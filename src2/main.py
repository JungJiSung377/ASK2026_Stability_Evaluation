"""
main.py — CO2 지중저장 안정성 평가 파이프라인 (PINN -> PCE -> PPO)

실행
----
    python main.py                      # 전체 (FDM 약 45분 + PINN 학습 + RL)
    python main.py --quick              # 축소 설정 (배관 점검용, 논문 수치 아님)
    python main.py --steps fdm          # 1단계만: 유한차분 기준해 데이터셋 생성
    python main.py --steps train        # 2단계만: PINN 학습 (기준해 캐시 필요)
    python main.py --steps analyze      # 3단계만: 검증/PCE/RL/XAI/그림 (pinn.pt 필요)

각 단계는 outputs/ 에 산출물을 저장하고 다음 단계가 이어받으므로,
오래 걸리는 FDM 과 PINN 학습을 따로 돌린 뒤 분석만 반복할 수 있다.

원본 코드 대비 수정 내역 (감사 태그)
------------------------------------
[C1]  검증 데이터를 모델 자신의 예측 + 잡음으로 만들던 순환논증을 제거.
      독립적인 유한차분 기준해(FDMReference)를 참값으로 삼고, 학습에 쓰지 않은
      (phi, eps) 조합에서 진짜 블라인드 테스트를 수행한다.
[C2]  PINN 이 스칼라 하나만 PCE 에 넘기던 단절을 제거. 파라미터화 PINN 이
      표본마다 서로 다른 압력장을 내놓고, PCE 는 그 PINN 을 대리모델링한다.
[C3]  L_data / L_BC 손실 추가(IC 는 신경망 구조에 하드 제약으로 내장).
[C4]  cp.E / cp.Std 정식 API 사용 + normed=True 전개계수로 교차검증.
      원본의 sqrt(sum(coefficients[1:]**2)) 는 직교기저 노름을 무시해
      표준편차를 약 10.7배 과대 계상했다.
[C5]  안전 제약은 학습 손실이 아니라 강화학습 보상에서만 부과.
[C6]  물성-압력 부호를 손수 쓴 수식이 아니라 물리(FDM/PINN)가 결정하게 함.
[C7]  행동이 상태에 누적 반영되는 MDP 환경 사용.
[M1]  학습 콜로케이션 시간 도메인을 추론 구간과 일치시킴 (외삽 제거).
[M6]  Sobol 1차/전체 민감도 지수를 실제로 계산.
[M7]  PCE 대리모델을 20,000회 몬테카를로 재표집해 경험적 분위수 사용.
[M11] 고정 주입 baseline 비교군 추가 + '위험률'을 명시적으로 정의.
[M12] R2 를 독립 참값 대비로 실제 계산.
[M13] IG 기여도를 집계·출력하고 완비성 공리를 검산.
[M14] FDM 대비 속도를 하드코딩 문자열이 아니라 실측.
[M15] PPO 시드 고정 + 다중 시드 반복 + Monitor 래퍼.
[m6]  루프 내 상수 재계산 제거(표본 전체를 한 번에 통과).
[m11] 미사용 import / 사장 변수 제거.
[m12] 그림 1~5 생성 코드 포함.
[m14] 학습량을 데모 수준(601 epoch)에서 실사용 수준으로 상향(설정 가능).
[m15] 콜로케이션 점을 매 스텝 재표집.
[m16] 미니배치 콜로케이션으로 메모리 위험 제거.
[m17] 정규성 가정(mu + 1.96 sigma) 대신 경험적 분위수 사용.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import chaospy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from environment import (
    MPA,
    CCSInjectionEnv,
    FDMReference,
    Realization,
    ReservoirConfig,
    basic_stats,
    create_geological_model,
    source_normalizer,
)
from models import (
    HeterogeneousField,
    Normalizer,
    PressurePINN,
    aggregate_attributions,
    compute_boundary_residual,
    compute_data_residual,
    compute_integrated_gradients,
    compute_pde_residual,
    safety_diagnostic,
)

OUT = Path("outputs")


# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    seed: int = 42
    n_train_designs: int = 12          # FDM 기준해를 푸는 학습용 (phi, eps) 조합 수
    n_blind_designs: int = 6           # 학습에 쓰지 않는 블라인드 검증 조합 수
    pinn_iters: int = 20000            # [m14] 원본 601 -> 실사용 수준
    pinn_batch: int = 4096             # [m16] 미니배치 콜로케이션
    pce_order: int = 3
    pce_train_samples: int = 80
    mc_samples: int = 20000            # [M7] 논문이 주장한 2만 회를 실제로 수행
    rl_timesteps: int = 200000
    rl_seeds: tuple = (0, 1, 2, 3, 4)  # [M15] 다중 시드
    weights: tuple = (0.1, 0.5, 0.9)
    n_risk_episodes: int = 200         # '위험률' 산정용 평가 에피소드 수
    phi_mean_mu: float = 0.189
    phi_mean_sd: float = 0.015
    eps_mu: float = 0.0
    # sd=0.30 은 k 를 20배 범위로 흩뜨려 저투과 실현에서 62 MPa(파쇄압 한참 초과)가
    # 나온다. 실측 KC 산포에 가까운 0.22 로 좁힌다.
    eps_sd: float = 0.22

    # 관측(모니터링 정) 설계 — [C3] L_data 용
    # 무작위 배치는 12.8 km 도메인 전체에 흩어져 압력 상승이 거의 없는 원거리만
    # 관측하게 된다(실측: 최근접 정도 2.7 km). 실제 모니터링 배열처럼 주입정에서
    # 반경을 늘려가며 배치하고, 각 정에서 세 깊이를 계측한다.
    monitor_offsets_cells: tuple = ((0, 0), (2, 0), (0, 4), (8, 8), (16, 0))
    monitor_depth_frac: tuple = (0.03, 0.72, 0.94)   # 덮개암 하부 / 개공 중앙 / 저부
    n_monitor_times: int = 12

    # 안전 평가 지점의 깊이(개공 구간 중앙). run_pce 와 진단 코드가 공유한다.
    safety_depth_frac: float = 0.72

    # 콜로케이션 표집에서 주입정 근방에 배정할 비율 [핵심]
    # 균일 표집만 쓰면 배치 4,096점 중 14점만 소스 영역에 들어가, 신경망이
    # '어디서나 0' 을 예측하는 것이 손실 최소해가 된다(실측: 블라인드 R2 = -0.19).
    near_well_fraction: float = 0.5


def reference_cache_key(cfg: ReservoirConfig, pc: PipelineConfig, years: np.ndarray) -> str:
    """기준해 캐시의 유효성 키. 물리 상수·설계 파라미터가 하나라도 바뀌면 달라진다."""
    payload = {
        "cfg": asdict(cfg),
        "pc": {k: getattr(pc, k) for k in
               ("n_train_designs", "n_blind_designs", "phi_mean_mu", "phi_mean_sd",
                "eps_mu", "eps_sd")},
        "years": np.round(np.asarray(years, dtype=float), 9).tolist(),
    }
    blob = json.dumps(payload, sort_keys=True, default=float).encode()
    return hashlib.sha1(blob).hexdigest()


def set_seed(seed: int) -> None:
    """[M15] 원본은 SB3 시드를 고정하지 않아 강화학습 결과가 재현되지 않았다."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def output_years(cfg: ReservoirConfig) -> np.ndarray:
    return np.unique(np.concatenate([
        np.array([0.0]),
        np.geomspace(0.25, cfg.inj_years, 18),
        np.linspace(cfg.inj_years, cfg.horizon_years, 15)]))


# ---------------------------------------------------------------------------
# STEP 1-2 : 실험 설계 + 유한차분 기준해 [C1]
# ---------------------------------------------------------------------------
def build_reference_dataset(cfg: ReservoirConfig, pc: PipelineConfig, years: np.ndarray):
    """(phi_mean, eps) 설계점마다 FDM 기준해를 풀어 '참값' 데이터셋을 만든다.

    학습용과 블라인드용을 **완전히 분리**해, 블라인드 집합은 PINN 학습의
    어떤 단계에서도 사용하지 않는다. 이것이 원본 C1(순환논증)의 근본 해결책이다.
    """
    cache = OUT / "fdm_reference.npz"
    key = reference_cache_key(cfg, pc, years)
    if cache.exists():
        d = np.load(cache)
        # 설정이 바뀌었는데 낡은 캐시를 쓰면 조용히 잘못된 결과가 나온다.
        # 격자 크기만 보는 부분 검사로는 mu, k_ref, 주입량, 분포 파라미터 변경을
        # 놓치므로 전체 설정의 해시를 키로 쓴다.
        if str(d["key"]) == key:
            print(f"  기준해 캐시 로드: {cache}")
            fields = d["fields"]      # NpzFile 은 접근할 때마다 압축을 풀므로 한 번만 읽는다
            return d["designs"], fields, d["k_star"], d["years"], int(d["n_train"])
        print(f"  기준해 캐시가 현재 설정과 불일치 -> 재계산합니다 ({cache})")

    dist = cp.J(cp.Normal(pc.phi_mean_mu, pc.phi_mean_sd), cp.Normal(pc.eps_mu, pc.eps_sd))
    n_total = pc.n_train_designs + pc.n_blind_designs
    designs = dist.sample(n_total, rule="latin_hypercube").T  # (n_total, 2)

    fields, k_stack = [], []
    j30 = int(np.argmin(np.abs(years - cfg.inj_years)))
    iz_s = int(round(pc.safety_depth_frac * (cfg.nz - 1)))
    for i, (phi_m, eps) in enumerate(designs):
        # 시드를 설계점마다 바꾸면 k* 가 (phi, eps) 의 함수가 아니게 되어,
        # PINN 입력에 없는 '실현 잡음'이 목표값에 섞인다. 모든 설계점이 같은
        # 공간 상관 패턴을 공유하도록 시드를 고정한다.
        geo = create_geological_model(
            cfg, phi_mean=float(phi_m), kc_residual_log10=float(eps),
            rng=np.random.default_rng(1000),
        )
        t0 = time.time()
        snaps = FDMReference(cfg, geo).solve(years)
        fields.append(snaps.astype(np.float32))
        k_stack.append(geo.k_star.astype(np.float32))
        tag = "train" if i < pc.n_train_designs else "BLIND"
        p_mon = cfg.to_mpa(snaps[j30, cfg.nx // 2, cfg.ny // 2, iz_s])
        print(f"  [{i+1:2d}/{n_total}] {tag:<5} phi={phi_m:.4f} eps={eps:+.3f} "
              f"k_med={np.median(geo.permeability_mD):6.2f} mD  "
              f"p_max={cfg.to_mpa(snaps[j30].max()):6.2f}  "
              f"p_안전지점={p_mon:6.2f} MPa  ({time.time()-t0:.0f}s)")

    fields = np.stack(fields)       # (n_total, n_years, nx, ny, nz)
    k_star = np.stack(k_stack)      # (n_total, nx, ny, nz)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(cache, designs=designs, fields=fields, k_star=k_star,
                        years=years, n_train=pc.n_train_designs, key=key)
    print(f"  -> 저장: {cache}")
    return designs, fields, k_star, years, pc.n_train_designs


# ---------------------------------------------------------------------------
# STEP 3 : 파라미터화 PINN 학습 [C2][C3][M1][m15][m16]
# ---------------------------------------------------------------------------
def sample_collocation_space(n, cfg, pc, device):
    """공간 콜로케이션 점 표집 — 절반은 균일, 절반은 주입정 근방에 집중.

    왜 필요한가
    -----------
    주입정 소스가 0 이 아닌 영역은 전체 격자의 약 0.35 % 뿐이다. 균일 표집을 쓰면
    배치 4,096 점 중 14 점만 그 영역에 들어가고, 나머지 99.7 % 는 p ~ 0, s = 0 인
    곳이다. 그러면 '어디서나 0 을 예측' 하는 것이 PDE 손실의 최소해가 되어 신경망이
    압력 상승 자체를 학습하지 않는다(실측: 블라인드 R2 = -0.19, 압력 6.2배 과소예측).

    반경을 로그 균일로 뽑는 이유는 방사류 압력장이 ln(r) 로 변하기 때문이다.

    주의: 이는 PDE 잔차를 균일 측도가 아니라 '주입정 중심 측도' 아래에서
    최소화한다는 뜻이다. 근정 영역의 정확도를 의도적으로 우선하는 선택이며,
    잔차 자체의 정의를 바꾸지는 않는다.
    """
    n_well = int(round(n * pc.near_well_fraction))
    n_uni = n - n_well
    parts = []
    if n_uni > 0:
        parts.append(torch.rand(n_uni, 3, device=device) * 2.0 - 1.0)
    if n_well > 0:
        r_min, r_max = 1.0 / cfg.nx, 0.6                 # 반 셀 ~ 약 3.8 km
        u = torch.rand(n_well, device=device)
        r = r_min * (r_max / r_min) ** u                 # 로그 균일
        th = torch.rand(n_well, device=device) * (2.0 * np.pi)
        # 주입정 중심은 정규화 좌표의 원점(xi = eta = 0)
        xi = (r * torch.cos(th)).clamp(-1.0, 1.0)
        eta = (r * torch.sin(th)).clamp(-1.0, 1.0)
        # 개공 구간(zeta 약 0.13~0.83)과 그 위 덮개암까지 덮도록 넓게 표집
        zeta = (torch.rand(n_well, device=device) * 1.3 - 0.3).clamp(-1.0, 1.0)
        parts.append(torch.stack([xi, eta, zeta], dim=1))
    return torch.cat(parts, dim=0)


def make_normalizer(cfg, pc) -> Normalizer:
    return Normalizer(
        cfg,
        phi_range=(pc.phi_mean_mu - 4 * pc.phi_mean_sd, pc.phi_mean_mu + 4 * pc.phi_mean_sd),
        eps_range=(pc.eps_mu - 4 * pc.eps_sd, pc.eps_mu + 4 * pc.eps_sd),
    )


def train_pinn(cfg, pc, designs, fields, k_star, years, n_train, device):
    norm = make_normalizer(cfg, pc)
    field = HeterogeneousField(cfg, k_star[:n_train], device)
    model = PressurePINN(cfg).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(pc.pinn_iters, 1), eta_min=1e-5)
    norm_const = source_normalizer(cfg)

    phi_t = torch.tensor(designs[:n_train, 0], dtype=torch.float32, device=device)
    eps_t = torch.tensor(designs[:n_train, 1], dtype=torch.float32, device=device)

    # --- [C3] 희소 모니터링 정 관측치 (FDM 참값에서 추출) ---
    rng = np.random.default_rng(pc.seed)
    depths = [int(round(f * (cfg.nz - 1))) for f in pc.monitor_depth_frac]
    well_cells = []
    for ox, oy in pc.monitor_offsets_cells:
        wxx = int(np.clip(cfg.nx // 2 + ox, 0, cfg.nx - 1))
        wyy = int(np.clip(cfg.ny // 2 + oy, 0, cfg.ny - 1))
        for dz_i in depths:
            well_cells.append((wxx, wyy, dz_i))
    t_idx = rng.choice(len(years), pc.n_monitor_times, replace=False)
    obs_r, obs_xyz, obs_t, obs_p = [], [], [], []
    for r in range(n_train):
        for (wxx, wyy, wzz) in well_cells:
            for ti in t_idx:
                obs_r.append(r)
                obs_xyz.append((wxx, wyy, wzz))
                obs_t.append(years[ti])
                obs_p.append(fields[r, ti, wxx, wyy, wzz])
    obs_xyz = np.array(obs_xyz, dtype=np.float32)
    xi_o, eta_o, zeta_o = norm.index_to_xi(obs_xyz[:, 0], obs_xyz[:, 1], obs_xyz[:, 2])
    data_coords = torch.tensor(
        np.stack([xi_o, eta_o, zeta_o, norm.year_to_tau(np.array(obs_t))], 1),
        dtype=torch.float32, device=device)
    data_r = torch.tensor(obs_r, dtype=torch.long, device=device)
    data_p = torch.tensor(np.array(obs_p)[:, None], dtype=torch.float32, device=device)
    radii_km = [np.hypot(ox, oy) * cfg.dx / 1000.0 for ox, oy in pc.monitor_offsets_cells]
    print(f"  관측 데이터 {data_p.shape[0]:,}점 "
          f"({len(pc.monitor_offsets_cells)} 정 x {len(depths)} 깊이 x "
          f"{pc.n_monitor_times} 시점 x {n_train} 실현)")
    print(f"  관측정 반경 [km] = {[round(r, 2) for r in radii_km]}")

    t_max_tau = float(norm.year_to_tau(years.max()))
    history = []
    t0 = time.time()
    for it in range(pc.pinn_iters + 1):
        opt.zero_grad(set_to_none=True)

        # --- [m15] 매 스텝 콜로케이션 재표집, [M1] 추론 구간 전체를 덮음 ---
        b = pc.pinn_batch
        r_idx = torch.randint(0, n_train, (b,), device=device)
        coll = torch.empty(b, 4, device=device)
        coll[:, :3] = sample_collocation_space(b, cfg, pc, device)
        coll[:, 3] = torch.rand(b, device=device) * t_max_tau
        res_pde = compute_pde_residual(
            model, norm, field, r_idx, coll,
            phi_t[r_idx].unsqueeze(1), eps_t[r_idx].unsqueeze(1), norm_const)
        loss_pde = (res_pde**2).mean()

        # --- [C3] 경계조건 ---
        nb = max(b // 4, 1)
        rb = torch.randint(0, n_train, (nb,), device=device)
        bc = torch.rand(nb, 4, device=device)
        bc[:, :3] = bc[:, :3] * 2.0 - 1.0
        bc[:, 3] = bc[:, 3] * t_max_tau
        axis = torch.randint(0, 3, (nb,), device=device)
        side = torch.randint(0, 2, (nb,), device=device).float() * 2.0 - 1.0
        bc[torch.arange(nb, device=device), axis] = side
        res_bc = compute_boundary_residual(
            model, norm, bc, phi_t[rb].unsqueeze(1), eps_t[rb].unsqueeze(1),
            axis, axis < 2)
        loss_bc = (res_bc**2).mean()

        # --- [C3] 관측 데이터 ---
        sel = torch.randint(0, data_p.shape[0], (min(b, data_p.shape[0]),), device=device)
        res_data = compute_data_residual(
            model, norm, data_coords[sel],
            phi_t[data_r[sel]].unsqueeze(1), eps_t[data_r[sel]].unsqueeze(1), data_p[sel])
        loss_data = (res_data**2).mean()

        # [C5] 안전 항은 학습 손실에 넣지 않는다.
        total = 1.0 * loss_data + 1.0 * loss_pde + 1.0 * loss_bc
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if it % max(pc.pinn_iters // 20, 1) == 0 or it == pc.pinn_iters:
            # float(tensor) 는 requires_grad 텐서에서 경고를 낸다. detach 후 스칼라화.
            history.append((it, loss_data.detach().item(), loss_pde.detach().item(),
                            loss_bc.detach().item()))
            print(f"  [{it:6d}/{pc.pinn_iters}] data {loss_data:.3e} | "
                  f"pde {loss_pde:.3e} | bc {loss_bc:.3e} | {time.time()-t0:.0f}s")
    return model, norm, np.array(history)


# ---------------------------------------------------------------------------
# STEP 4 : 블라인드 검증 [C1][M12]
# ---------------------------------------------------------------------------
def blind_validation(cfg, model, norm, designs, fields, years, n_train, device,
                     n_points=20000):
    """학습에 전혀 쓰이지 않은 (phi, eps) 조합의 FDM 참값 전체 장과 비교한다."""
    rng = np.random.default_rng(7)
    truth, pred, t_all = [], [], []
    for r in range(n_train, designs.shape[0]):
        ix = rng.integers(0, cfg.nx, n_points)
        iy = rng.integers(0, cfg.ny, n_points)
        iz = rng.integers(0, cfg.nz, n_points)
        ti = rng.integers(0, len(years), n_points)
        xi, eta, zeta = norm.index_to_xi(ix.astype(np.float32), iy.astype(np.float32),
                                         iz.astype(np.float32))
        coords = torch.tensor(
            np.stack([xi, eta, zeta, norm.year_to_tau(years[ti])], 1),
            dtype=torch.float32, device=device)
        phi = torch.full((n_points, 1), float(designs[r, 0]), device=device)
        eps = torch.full((n_points, 1), float(designs[r, 1]), device=device)
        phi_n, eps_n = norm.params_to_unit(phi, eps)
        with torch.no_grad():
            p = model(coords[:, [0]], coords[:, [1]], coords[:, [2]], coords[:, [3]],
                      phi_n, eps_n).cpu().numpy().ravel()
        pred.append(p)
        truth.append(fields[r, ti, ix, iy, iz])
        t_all.append(years[ti])
    truth = np.concatenate(truth).astype(np.float64)
    pred = np.concatenate(pred).astype(np.float64)
    t_all = np.concatenate(t_all)

    def metrics(mask):
        if mask.sum() < 10:
            return None
        rs = float(np.sqrt(mean_squared_error(truth[mask], pred[mask])))
        return {"rmse_star": rs, "rmse_mpa": rs * cfg.dp_ref / MPA,
                "r2": float(r2_score(truth[mask], pred[mask])), "n": int(mask.sum())}

    # 전체 격자를 균일 표집하면 압력이 사실상 0 인 영역(원거리·감압 후)이 표본을
    # 지배해 "0 을 0 으로 예측"하는 능력만 측정하게 된다. 계층별로 함께 보고한다.
    thresh = 0.05 * np.abs(truth).max()
    strata = {
        "all": np.ones_like(truth, dtype=bool),
        "injection_period": t_all <= cfg.inj_years,
        "pressurized": np.abs(truth) > thresh,
        "pressurized_injection": (np.abs(truth) > thresh) & (t_all <= cfg.inj_years),
    }
    out = {k: metrics(m) for k, m in strata.items()}
    ref = out["pressurized"] or out["all"]

    # [C5] 안전 항을 학습 손실에서 뺐다는 주장의 실증. 손실에 넣어 훈련하면 모델이
    # 임계압력 초과를 계통적으로 과소 예측하게 된다. 참값과 예측의 초과 통계를
    # 나란히 보고해 그런 편향이 없는지 확인한다.
    exc_t = safety_diagnostic(torch.tensor(truth, dtype=torch.float32), cfg).numpy()
    exc_p = safety_diagnostic(torch.tensor(pred, dtype=torch.float32), cfg).numpy()
    safety = {
        "truth_exceed_frac": float((exc_t > 0).mean()),
        "pred_exceed_frac": float((exc_p > 0).mean()),
        "truth_max_exceed_mpa": float(exc_t.max() * cfg.dp_ref / MPA),
        "pred_max_exceed_mpa": float(exc_p.max() * cfg.dp_ref / MPA),
    }
    return {
        "rmse_mpa": ref["rmse_mpa"], "rmse_star": ref["rmse_star"],
        "r2": ref["r2"], "n": ref["n"],
        "strata": out, "safety": safety, "truth": truth, "pred": pred,
    }


def check_surrogate_bias(cfg, pc, model, norm, designs, fields, years, n_train, device):
    """PINN 이 '안전 평가 지점'에서 FDM 참값을 재현하는지 직접 대조한다.

    왜 별도 점검이 필요한가
    -----------------------
    전역 RMSE 는 압력이 거의 0 인 표본이 대다수라 근정 과압을 6배 과소예측해도
    작게 나온다. 그러면 PCE 평균이 낮아지고, 강화학습 안전 제약이 아예 구속되지
    않아 파레토 스캔이 공허해진다. 그 증상은 run_rl 의 assertion 이 잡지만,
    원인을 지목하려면 이 지점 대조가 필요하다.
    """
    iz = int(round(pc.safety_depth_frac * (cfg.nz - 1)))
    ix, iy = cfg.nx // 2, cfg.ny // 2
    xi, eta, zeta = norm.index_to_xi(np.float32(ix), np.float32(iy), np.float32(iz))
    j30 = int(np.argmin(np.abs(years - cfg.inj_years)))
    rows = []
    for r in range(n_train, designs.shape[0]):
        tau = torch.tensor(norm.year_to_tau(years), dtype=torch.float32,
                           device=device).view(-1, 1)
        m = tau.shape[0]
        base = torch.tensor([[xi, eta, zeta]], dtype=torch.float32, device=device).repeat(m, 1)
        phi = torch.full((m, 1), float(designs[r, 0]), device=device)
        eps = torch.full((m, 1), float(designs[r, 1]), device=device)
        phi_n, eps_n = norm.params_to_unit(phi, eps)
        with torch.no_grad():
            pred = model(base[:, [0]], base[:, [1]], base[:, [2]], tau,
                         phi_n, eps_n).cpu().numpy().ravel()
        truth = fields[r, :, ix, iy, iz]
        rows.append({
            "design": int(r), "phi": float(designs[r, 0]), "eps": float(designs[r, 1]),
            "fdm_30yr_mpa": float(cfg.to_mpa(truth[j30])),
            "pinn_30yr_mpa": float(cfg.to_mpa(pred[j30])),
            "overpressure_ratio": float(pred[j30] / max(truth[j30], 1e-9)),
        })
    ratios = np.array([x["overpressure_ratio"] for x in rows])
    return {"per_design": rows, "mean_ratio": float(ratios.mean()),
            "min_ratio": float(ratios.min()), "max_ratio": float(ratios.max())}


# ---------------------------------------------------------------------------
# STEP 5 : PCE + Sobol [C4][M6][M7]
# ---------------------------------------------------------------------------
def run_pce(cfg, pc, model, norm, years, device):
    dist = cp.J(cp.Normal(pc.phi_mean_mu, pc.phi_mean_sd), cp.Normal(pc.eps_mu, pc.eps_sd))
    # [C4] normed=True 로 정규직교 기저를 만들고, 전개계수를 retall 로 직접 받는다.
    expansion = cp.generate_expansion(pc.pce_order, dist, normed=True)
    train = dist.sample(pc.pce_train_samples, rule="latin_hypercube")

    # --- 응답: PINN 이 예측한 '주입 개공 구간 중앙'의 압력 ---
    # 안전 기준 45 MPa = 1.5 x 초기압은 파쇄압의 보수적 대용치이므로, 압력이 가장
    # 높은 개공 구간에서 평가해야 한다. 덮개암 상단(iz=0)에서 평가하면 kv/kh=0.1
    # 때문에 압력이 훨씬 낮아 임계값에 도달조차 하지 않고, 그러면 강화학습의
    # 안전 항이 항등적으로 0 이 되어 가중치 스캔이 무의미해진다.
    # [m6] 표본 전체를 한 번에 신경망에 통과시킨다(원본은 루프 안에서 루프와
    #      무관한 상수를 60회 재계산했다).
    def response(samples):
        n = samples.shape[1]
        xi, eta, zeta = norm.index_to_xi(
            np.full(n, cfg.nx / 2.0 - 0.5, np.float32),
            np.full(n, cfg.ny / 2.0 - 0.5, np.float32),
            np.full(n, pc.safety_depth_frac * (cfg.nz - 1), np.float32))
        out = np.zeros((n, len(years)))
        phi = torch.tensor(samples[0], dtype=torch.float32, device=device).unsqueeze(1)
        eps = torch.tensor(samples[1], dtype=torch.float32, device=device).unsqueeze(1)
        phi_n, eps_n = norm.params_to_unit(phi, eps)
        base = torch.tensor(np.stack([xi, eta, zeta], 1), dtype=torch.float32, device=device)
        with torch.no_grad():
            for j, yr in enumerate(years):
                tau = torch.full((n, 1), float(norm.year_to_tau(yr)), device=device)
                out[:, j] = model(base[:, [0]], base[:, [1]], base[:, [2]], tau,
                                  phi_n, eps_n).cpu().numpy().ravel()
        return out

    y_train = response(train)
    mc = dist.sample(pc.mc_samples, rule="sobol")          # [M7] 실제 2만 회

    means, stds, p975, s1, st = [], [], [], [], []
    for j in range(len(years)):
        surrogate, coeffs = cp.fit_regression(expansion, train, y_train[:, j], retall=1)

        # --- [C4] 두 경로(정식 API vs 정규직교 전개계수)를 평균·표준편차 모두 교차검증 ---
        m_api, s_api = float(cp.E(surrogate, dist)), float(cp.Std(surrogate, dist))
        m_coef = float(coeffs[0])                           # 정규직교 기저이므로 c0 = 평균
        s_coef = float(np.sqrt(np.sum(np.asarray(coeffs[1:]) ** 2)))
        if abs(m_api - m_coef) > 1e-3 * max(abs(m_api), 1e-9):
            raise AssertionError(
                f"PCE 평균 불일치: API {m_api:.6f} vs 계수 {m_coef:.6f} (t={years[j]}). "
                f"기저가 정규직교가 아니거나 상수항이 첫 번째가 아닙니다.")
        if abs(s_api - s_coef) > 1e-3 * max(s_api, 1e-9):
            raise AssertionError(
                f"PCE 표준편차 불일치: API {s_api:.6f} vs 계수 {s_coef:.6f} (t={years[j]})")

        # [M7][m17] 경험적 분위수. mu + 1.96*sigma 는 응답이 정규분포일 때만
        # 정확한데, PCE 대리모델은 일반적으로 비정규이므로 재표집 분위수를 쓴다.
        y_mc = surrogate(*mc)
        means.append(m_api)
        stds.append(s_api)
        p975.append(float(np.quantile(y_mc, 0.975)))
        # [M6] Sobol 민감도 — 입력이 독립이므로 분산분해가 정의된다
        s1.append(np.asarray(cp.Sens_m(surrogate, dist), dtype=float))
        st.append(np.asarray(cp.Sens_t(surrogate, dist), dtype=float))

    return {
        "years": years, "mean": np.array(means), "std": np.array(stds),
        "p975": np.array(p975), "sobol_first": np.array(s1), "sobol_total": np.array(st),
        "dist": dist, "surrogate_response": response,
    }


# ---------------------------------------------------------------------------
# STEP 6 : 강화학습 + 고정주입 baseline [C7][M10][M11][M15]
# ---------------------------------------------------------------------------
def make_sampler(cfg, pc, pce, sigma_star):
    """PCE 대리모델에서 에피소드별 실현을 표집한다. [m9]"""
    years = pce["years"]
    n_years = int(cfg.inj_years)
    grid = np.arange(n_years + 1, dtype=float)
    resp = pce["surrogate_response"]

    def sampler(np_random: np.random.Generator) -> Realization:
        phi = np_random.normal(pc.phi_mean_mu, pc.phi_mean_sd)
        eps = np_random.normal(pc.eps_mu, pc.eps_sd)
        curve = resp(np.array([[phi], [eps]]))[0]
        p_base = np.interp(grid, years, curve)
        # 주입 기간 중 과압은 물리적으로 단조 증가한다. 대리모델의 수치 잡음이
        # b(t) < 0 (주입을 늘렸는데 압력이 내려감) 을 만들지 않도록 하는 가드.
        p_base = np.maximum.accumulate(np.maximum(p_base - p_base[0], 0.0))
        # 잔여 모델 불확실성: 검증된 블라인드 오차가 예측 지평에 따라 완만히 증가
        sigma = sigma_star * (1.0 + grid / n_years)
        return Realization(p_base=p_base, sigma=sigma, phi_mean=float(phi),
                           kc_residual=float(eps))

    return sampler


def evaluate_policy(env, policy, n_episodes, seed0=10_000):
    """policy(obs) -> action.  '위험률' = 임계압력을 한 번이라도 초과한 에피소드 비율. [M11]"""
    cfg = env.cfg
    peaks, revenues, breaches = [], [], 0
    for e in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + e)
        peak, rev, over = cfg.p_init / MPA, 0.0, False
        while True:
            obs, r, term, trunc, info = env.step(policy(obs))
            peak = max(peak, info["p975_mpa"])
            rev += info["revenue_usd_per_t"]
            if info["p975_mpa"] > cfg.p_limit / MPA:
                over = True
            if term or trunc:
                break
        peaks.append(peak)
        revenues.append(rev)
        breaches += int(over)
    return {
        "peak_p975_mean": float(np.mean(peaks)),
        "peak_p975_p95": float(np.quantile(peaks, 0.95)),
        "peak_p975_max": float(np.max(peaks)),
        "revenue_mean": float(np.mean(revenues)),
        "risk_rate": breaches / n_episodes,     # 명시적으로 정의된 '위험률'
    }


def run_rl(cfg, pc, sampler):
    results = {"baselines": {}, "ppo": {}}

    # --- [M11] 고정 주입 baseline ---
    for a_fixed in (1.0, 1.5):
        env = CCSInjectionEnv(cfg, sampler, weight_econ=0.5)
        raw = np.array([(a_fixed - 1.0) / 0.5], dtype=np.float32)
        results["baselines"][f"fixed_{a_fixed}"] = evaluate_policy(
            env, lambda obs, a=raw: a, pc.n_risk_episodes)

    # 안전 제약이 도달 가능한지 먼저 확인한다. 최대 주입에서도 임계압력을 한 번도
    # 넘지 않으면 위험률이 모든 정책에서 0 이 되어 파레토 비교가 공허해진다.
    aggressive = results["baselines"]["fixed_1.5"]
    if aggressive["risk_rate"] <= 0.0:
        raise AssertionError(
            f"최대 주입(a=1.5)에서도 임계압력 초과가 발생하지 않습니다 "
            f"(최대 P97.5 = {aggressive['peak_p975_max']:.2f} MPa < "
            f"{cfg.p_limit/MPA:.1f} MPa). 안전 제약이 비활성이므로 가중치 스캔이 "
            f"무의미합니다. 먼저 Step 4b 의 과압 재현비를 확인하고, 정상이라면 "
            f"inject_mt_per_year 를 올리거나 p_limit 을 조정하십시오.")

    # --- PPO ---
    env0 = CCSInjectionEnv(cfg, sampler, weight_econ=0.5)
    check_env(env0, warn=True)          # SB3 규약 검사 (원본에는 없었음)

    for w in pc.weights:
        per_seed = []
        for sd in pc.rl_seeds:
            env = Monitor(CCSInjectionEnv(cfg, sampler, weight_econ=w))
            env.reset(seed=sd)
            env.action_space.seed(sd)
            agent = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4,
                        n_steps=2048, batch_size=256, seed=sd)   # [M15] 시드 고정
            agent.learn(total_timesteps=pc.rl_timesteps)
            ev = CCSInjectionEnv(cfg, sampler, weight_econ=w)
            per_seed.append(evaluate_policy(
                ev, lambda o, m=agent: m.predict(o, deterministic=True)[0],
                pc.n_risk_episodes))
            print(f"    w={w} seed={sd}: risk={per_seed[-1]['risk_rate']:.3f} "
                  f"peak={per_seed[-1]['peak_p975_mean']:.2f} MPa "
                  f"rev={per_seed[-1]['revenue_mean']:.0f}")
        results["ppo"][w] = {
            k: {"mean": float(np.mean([p[k] for p in per_seed])),
                "std": float(np.std([p[k] for p in per_seed]))}
            for k in per_seed[0]
        }
    return results


# ---------------------------------------------------------------------------
# STEP 7 : XAI [M13][m13]
# ---------------------------------------------------------------------------
def run_xai(cfg, pc, model, norm, device, year=10.0, n_global=20000):
    """[M13] 기여도를 실제로 집계·출력하고 완비성 공리를 검산한다.

    기준점은 '주입 이전(t=0), 도메인 중앙, 평균 물성' = 정규화 좌표의 원점으로,
    물리적으로 무정보 상태에 해당한다(원본은 도메인 모서리를 썼다).
    """
    base = torch.zeros(1, 6, device=device)

    # --- (1) 전역 집계: 6개 입력을 모두 각자의 범위 전체에서 표집 ---
    # phi, eps 를 평균값으로 고정하면 정규화 후 정확히 0 이 되어 기준점과 같아지고,
    # 기여도가 항등적으로 0 으로 나온다. 반드시 함께 변화시켜야 한다.
    rng = np.random.default_rng(pc.seed)
    g = rng.uniform(-1.0, 1.0, size=(n_global, 6)).astype(np.float32)
    g[:, 3] = rng.uniform(0.0, float(norm.year_to_tau(cfg.horizon_years)), n_global)
    g_t = torch.tensor(g, dtype=torch.float32, device=device)
    attr_g, gap = compute_integrated_gradients(model, g_t, base, steps=64)
    agg = aggregate_attributions(attr_g, delta=g_t - base)

    # --- (2) 시각화용 히트맵: 개공 구간 깊이 슬라이스, 지정 시점 ---
    n = cfg.nx * cfg.ny
    ix, iy = np.meshgrid(np.arange(cfg.nx), np.arange(cfg.ny), indexing="ij")
    xi, eta, zeta = norm.index_to_xi(
        ix.ravel().astype(np.float32), iy.ravel().astype(np.float32),
        np.full(n, pc.safety_depth_frac * (cfg.nz - 1), np.float32))
    phi_n, eps_n = norm.params_to_unit(
        np.full((n, 1), pc.phi_mean_mu, np.float32), np.full((n, 1), pc.eps_mu, np.float32))
    slab = torch.tensor(
        np.concatenate([np.stack([xi, eta, zeta,
                                  np.full(n, norm.year_to_tau(year), np.float32)], 1),
                        phi_n, eps_n], 1), dtype=torch.float32, device=device)
    attr_slab, _ = compute_integrated_gradients(model, slab, base, steps=32)

    # [M13] 소스항은 x, y 에 완전 대칭이므로 물리적으로는 기여도도 대칭이어야 한다.
    # 비대칭이 크게 나온다면 그것은 '모델이 이방성을 학습했다'는 증거가 아니라
    # 가중치 초기화·표집 잡음의 증거로 읽어야 한다.
    sx, sy = agg["x"]["sensitivity_per_unit"], agg["y"]["sensitivity_per_unit"]
    sym = abs(sx - sy) / max(sx + sy, 1e-30)
    return {"aggregate": agg, "completeness_gap": gap, "xy_asymmetry": float(sym),
            "heatmap": attr_slab.abs().sum(1).detach().cpu().numpy().reshape(cfg.nx, cfg.ny)}


# ---------------------------------------------------------------------------
# STEP 8 : 실측 속도 벤치마크 [M14]
# ---------------------------------------------------------------------------
def benchmark(cfg, model, geo, device, n_eval=20000):
    """원본은 'FDM 대비 10,000배'를 문자열로 출력만 했다. 여기서는 실제로 잰다."""
    t0 = time.time()
    FDMReference(cfg, geo).solve(np.array([0.0, cfg.inj_years]))
    fdm_s = time.time() - t0

    xi = torch.rand(n_eval, 4, device=device) * 2 - 1
    # tau 는 [0,1] 구간에서만 정의된다. 음수 tau 를 넣으면 하드 초기조건의
    # exp(-t/tau_ic) 가 float32 에서 overflow 하여 출력이 -inf/NaN 이 된다.
    xi[:, 3] = torch.rand(n_eval, device=device)
    phi = torch.zeros(n_eval, 1, device=device)
    eps = torch.zeros(n_eval, 1, device=device)
    with torch.no_grad():                       # 워밍업
        model(xi[:, [0]], xi[:, [1]], xi[:, [2]], xi[:, [3]], phi, eps)
    if device.type == "cuda":
        torch.cuda.synchronize()                # GPU 비동기 커널 보정
    t0 = time.time()
    with torch.no_grad():
        model(xi[:, [0]], xi[:, [1]], xi[:, [2]], xi[:, [3]], phi, eps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    pinn_s = time.time() - t0
    return {"fdm_full_solve_s": fdm_s, "pinn_infer_s": pinn_s, "pinn_points": n_eval,
            "note": "PINN 추론은 1회 학습 비용(별도 계상)을 상각한 뒤의 질의 비용"}


# ---------------------------------------------------------------------------
# 그림 [m12]
# ---------------------------------------------------------------------------
def make_figures(cfg, geo, pce, blind, xai, rl, history):
    """그림 1~5 생성.  [m12] 원본에는 그림 생성 코드가 전혀 없어 재현이 불가능했다.

    라벨은 영문으로 둔다 — 한글 글꼴이 없는 환경에서 matplotlib 이 조용히
    두부(tofu) 상자를 그려 그림이 못 쓰게 되는 것을 피하기 위함이다.
    """
    OUT.mkdir(exist_ok=True)
    to_mpa = cfg.to_mpa

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    im0 = ax[0].imshow(geo.porosity[:, :, cfg.nz // 2].T, origin="lower", cmap="viridis")
    ax[0].set_title("Porosity (mid-depth slice)"); fig.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(np.log10(geo.permeability_mD[:, :, cfg.nz // 2]).T,
                       origin="lower", cmap="magma")
    ax[1].set_title("log10 permeability [mD]"); fig.colorbar(im1, ax=ax[1])
    ax[2].imshow(geo.seismic_time[:, cfg.ny // 2, :].T, origin="upper", cmap="gray_r",
                 aspect="auto", extent=[0, cfg.nx, geo.twt_axis[-1], 0])
    ax[2].set_title("Synthetic seismic (time domain)")
    ax[2].set_ylabel("TWT [s]"); ax[2].set_xlabel("inline")
    fig.tight_layout(); fig.savefig(OUT / "fig1_geology.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(pce["years"], to_mpa(pce["mean"]), label="PCE mean")
    ax.fill_between(pce["years"], to_mpa(pce["mean"] - 1.96 * pce["std"]),
                    to_mpa(pce["mean"] + 1.96 * pce["std"]), alpha=.2,
                    label="mean $\\pm$ 1.96$\\sigma$")
    ax.plot(pce["years"], to_mpa(pce["p975"]), "--", label="P97.5 (empirical MC quantile)")
    ax.axvline(cfg.inj_years, color="0.5", lw=.8)
    ax.axhline(cfg.p_limit / MPA, color="crimson", ls=":", label="operating limit 45 MPa")
    ax.set_xlabel("Year"); ax.set_ylabel("Pressure [MPa]"); ax.legend(fontsize=8)
    ax.set_title("Pressure forecast with geological uncertainty")
    fig.tight_layout(); fig.savefig(OUT / "fig2_pce.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, nm in enumerate([r"$\bar\phi$", r"KC residual $\varepsilon$"]):
        ax.plot(pce["years"], pce["sobol_first"][:, i], label=f"$S_1$ {nm}")
        ax.plot(pce["years"], pce["sobol_total"][:, i], "--", label=f"$S_T$ {nm}")
    ax.set_xlabel("Year"); ax.set_ylabel("Sobol index"); ax.legend(fontsize=8)
    ax.set_title("Global sensitivity (valid: inputs are independent)")
    fig.tight_layout(); fig.savefig(OUT / "fig3_sobol.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    names, risks, revs = [], [], []
    for k, v in rl["baselines"].items():
        names.append(k); risks.append(v["risk_rate"]); revs.append(v["revenue_mean"])
    for w, v in rl["ppo"].items():
        names.append(f"PPO w={w}"); risks.append(v["risk_rate"]["mean"])
        revs.append(v["revenue_mean"]["mean"])
    ax[0].scatter(revs, risks, s=45)
    for n_, x_, y_ in zip(names, revs, risks):
        ax[0].annotate(n_, (x_, y_), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax[0].set_xlabel("Cumulative revenue [USD/tCO2]")
    ax[0].set_ylabel("Risk rate  P(peak $P_{97.5}$ > 45 MPa)")
    ax[0].set_title("Pareto: revenue vs. limit-exceedance probability")
    ax[1].plot(history[:, 0], history[:, 1], label="data")
    ax[1].plot(history[:, 0], history[:, 2], label="PDE")
    ax[1].plot(history[:, 0], history[:, 3], label="BC")
    ax[1].set_yscale("log"); ax[1].set_xlabel("iteration"); ax[1].set_ylabel("MSE")
    ax[1].legend(); ax[1].set_title("PINN loss history")
    fig.tight_layout(); fig.savefig(OUT / "fig4_pareto_loss.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    sub = slice(None, None, max(1, blind["truth"].size // 4000))
    ax[0].scatter(to_mpa(blind["truth"][sub]), to_mpa(blind["pred"][sub]), s=3, alpha=.3)
    lim = [to_mpa(blind["truth"].min()), to_mpa(blind["truth"].max())]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set_xlabel("FDM reference [MPa]"); ax[0].set_ylabel("PINN prediction [MPa]")
    ax[0].set_title(f"Blind test (unseen geology)\n"
                    f"RMSE={blind['rmse_mpa']:.3f} MPa   $R^2$={blind['r2']:.4f}")
    im = ax[1].imshow(xai["heatmap"].T, origin="lower", cmap="inferno")
    ax[1].set_title("Integrated Gradients (perforation depth, t=10 yr)")
    fig.colorbar(im, ax=ax[1])
    fig.tight_layout(); fig.savefig(OUT / "fig5_validation_xai.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="CO2 지중저장 안정성 평가 파이프라인 (PINN -> PCE -> PPO)")
    # verify_fixes.py 는 --fast 를 쓴다. 두 스크립트에서 같은 플래그가 통하도록 별칭.
    ap.add_argument("--quick", "--fast", dest="quick", action="store_true",
                    help="축소 설정으로 빠르게 실행 (배관 점검용, 논문 수치 아님)")
    ap.add_argument("--steps", default="all",
                    choices=["all", "fdm", "train", "analyze"],
                    help="fdm=기준해만 / train=PINN 학습만 / analyze=검증·PCE·RL·그림만")
    args = ap.parse_args()

    cfg = ReservoirConfig()
    pc = PipelineConfig()
    if args.quick:
        pc.pinn_iters, pc.rl_timesteps = 4000, 20000
        pc.n_train_designs, pc.n_blind_designs = 6, 3
        pc.mc_samples, pc.n_risk_episodes = 20000, 50
        pc.rl_seeds = (0, 1)
        print("=" * 78)
        print("  [--quick] 배관 점검용 축소 설정입니다.")
        print("  학습 4,000회 / 설계점 9개 / 시드 2개 — PINN 이 수렴하지 않으므로")
        print("  여기서 나오는 압력·위험도·민감도 수치는 논문에 쓰면 안 됩니다.")
        print("  실제 수치는 --quick 없이 실행하십시오.")
        print("=" * 78)

    set_seed(pc.seed)
    OUT.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    years = output_years(cfg)
    print(f"장치 {device} | 도메인 {cfg.extent[0]/1000:.1f}x{cfg.extent[1]/1000:.1f} km "
          f"x {cfg.extent[2]:.0f} m | 주입 {cfg.inject_mt_per_year} Mt/yr | "
          f"출력 시점 {len(years)}개 | steps={args.steps}")

    # ---------------- Step 0-2 : 기준해 ----------------
    if args.steps in ("all", "fdm"):
        print("\nStep 0: 기준해 시간 수렴성 확인 ...")
        geo_probe = create_geological_model(cfg, pc.phi_mean_mu, pc.eps_mu,
                                            rng=np.random.default_rng(1000))
        conv = FDMReference(cfg, geo_probe).convergence_check(
            np.array([0.0, cfg.inj_years]), dt_scales=(1.0, 0.5))
        abs_change = abs(conv["peak_mpa"][0] - conv["peak_mpa"][-1])
        print(f"  dt 스케일 {conv['dt_scales']} -> 30년 최대압력 "
              f"{[round(v, 3) for v in conv['peak_mpa']]} MPa")
        print(f"  스텝을 절반으로 줄였을 때 변화: {abs_change:.3f} MPa "
              f"(과압 대비 {conv['rel_change']:.1%})")
        if conv["rel_change"] > 0.03:
            print("  [경고] 기준해가 시간 스텝에 충분히 수렴하지 않았습니다. "
                  "build_time_grid 를 더 세분하십시오.")
        (OUT / "convergence.json").write_text(json.dumps(conv, indent=2), encoding="utf-8")

        print("\nStep 1-2: 유한차분 기준해 데이터셋 생성 (독립적 참값) ...")
        designs, fields, k_star, years, n_train = build_reference_dataset(cfg, pc, years)
        if args.steps == "fdm":
            print("\n[--steps fdm] 기준해 생성 완료. 다음: python main.py --steps train")
            return
    else:
        # train / analyze 단계에서 캐시가 없으면 45분짜리 재계산이 조용히 시작된다.
        # 의도치 않은 대기를 막기 위해 먼저 확인하고 안내한다.
        if not (OUT / "fdm_reference.npz").exists():
            raise SystemExit(
                f"{OUT/'fdm_reference.npz'} 가 없습니다. "
                f"먼저 python main.py --steps fdm 을 실행하십시오.")
        print("\nStep 1-2: 기준해 데이터셋 로드 ...")
        designs, fields, k_star, years, n_train = build_reference_dataset(cfg, pc, years)

    # ---------------- Step 3 : PINN 학습 ----------------
    ckpt = OUT / "pinn.pt"
    if args.steps in ("all", "train"):
        print("\nStep 3: 파라미터화 PINN 학습 (data + PDE + BC, IC 는 구조적 하드 제약) ...")
        model, norm, history = train_pinn(cfg, pc, designs, fields, k_star, years,
                                          n_train, device)
        # 학습된 PINN 을 저장한다. 20,000회 학습 결과를 프로세스 종료와 함께 버리면
        # 그림 하나 다시 그리려고 전체를 재학습해야 한다.
        torch.save({"state_dict": model.state_dict(), "history": history,
                    "cfg": asdict(cfg), "pc": asdict(pc)}, ckpt)
        print(f"  -> 학습된 PINN 저장: {ckpt}")
        if args.steps == "train":
            print("\n[--steps train] 학습 완료. 다음: python main.py --steps analyze")
            return
    else:
        if not ckpt.exists():
            raise SystemExit(f"{ckpt} 가 없습니다. 먼저 python main.py --steps train 을 실행하십시오.")
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        model = PressurePINN(cfg).to(device)
        model.load_state_dict(blob["state_dict"])
        norm = make_normalizer(cfg, pc)
        history = np.asarray(blob["history"])
        print(f"\nStep 3: 저장된 PINN 로드 ({ckpt})")

    # ---------------- Step 4 이후 : 분석 ----------------
    print("\nStep 4: 블라인드 검증 (학습에 쓰지 않은 지질 조건) ...")
    blind = blind_validation(cfg, model, norm, designs, fields, years, n_train, device)
    for name, m in blind["strata"].items():
        if m:
            print(f"  [{name:<22}] RMSE {m['rmse_mpa']:.4f} MPa  "
                  f"R² {m['r2']:+.4f}  n={m['n']:,}")
    sf = blind["safety"]
    print(f"  임계압력 초과 비율: 참값(FDM) {sf['truth_exceed_frac']*100:.2f}% vs "
          f"PINN {sf['pred_exceed_frac']*100:.2f}%  "
          f"(최대 초과 {sf['truth_max_exceed_mpa']:.2f} vs {sf['pred_max_exceed_mpa']:.2f} MPa)")
    print("    -> 두 값이 비슷해야 함. PINN 쪽이 계통적으로 낮으면 안전 항이 "
          "물리해를 오염시키고 있다는 신호(원본 C5).")

    print("\nStep 4b: 안전 평가 지점에서 PINN vs FDM 직접 대조 ...")
    bias = check_surrogate_bias(cfg, pc, model, norm, designs, fields, years,
                                n_train, device)
    print(f"  {'설계':>5}{'phi':>8}{'eps':>8}{'FDM 30yr':>11}{'PINN 30yr':>12}{'과압비':>9}")
    for row in bias["per_design"]:
        print(f"  {row['design']:>5}{row['phi']:>8.4f}{row['eps']:>+8.3f}"
              f"{row['fdm_30yr_mpa']:>11.2f}{row['pinn_30yr_mpa']:>12.2f}"
              f"{row['overpressure_ratio']:>9.3f}")
    print(f"  과압 재현비 평균 {bias['mean_ratio']:.3f} "
          f"(범위 {bias['min_ratio']:.3f}~{bias['max_ratio']:.3f}, 1.0 이 이상적)")
    if not 0.7 <= bias["mean_ratio"] <= 1.4:
        print("  [경고] PINN 이 안전 평가 지점의 과압을 계통적으로 잘못 예측합니다. "
              "이대로면 PCE 위험도와 강화학습 제약이 모두 왜곡됩니다.")
        print("         pinn_iters 를 늘리거나 near_well_fraction 을 올리십시오.")

    print("\nStep 5: PCE 불확실성 정량화 + Sobol 민감도 ...")
    pce = run_pce(cfg, pc, model, norm, years, device)
    j30 = int(np.argmin(np.abs(years - cfg.inj_years)))
    print(f"  30년: 평균 {cfg.to_mpa(pce['mean'][j30]):.2f} MPa, "
          f"σ {pce['std'][j30]*cfg.dp_ref/MPA:.3f} MPa, "
          f"P97.5 {cfg.to_mpa(pce['p975'][j30]):.2f} MPa (경험적 분위수)")
    print(f"  Sobol S1 (phi, eps) = {np.round(pce['sobol_first'][j30], 3)}  "
          f"ST = {np.round(pce['sobol_total'][j30], 3)}")

    print("\nStep 6: 강화학습 제어 + 고정주입 baseline ...")
    rl = run_rl(cfg, pc, make_sampler(cfg, pc, pce, blind["rmse_star"]))
    for k, v in rl["baselines"].items():
        print(f"  baseline {k}: 위험률 {v['risk_rate']:.3f}  "
              f"최대 P97.5 평균 {v['peak_p975_mean']:.2f} MPa  수익 {v['revenue_mean']:.0f}")
    for w, v in rl["ppo"].items():
        print(f"  PPO w={w}: 위험률 {v['risk_rate']['mean']:.3f}±{v['risk_rate']['std']:.3f}  "
              f"최대 P97.5 {v['peak_p975_mean']['mean']:.2f} MPa  "
              f"수익 {v['revenue_mean']['mean']:.0f}")

    print("\nStep 7: XAI (Integrated Gradients) ...")
    xai = run_xai(cfg, pc, model, norm, device)
    print(f"  {'입력':<5}{'평균|기여도|':>14}{'raw 비중':>10}"
          f"{'단위입력당 민감도':>18}{'정규화 비중':>12}")
    for n_, v in xai["aggregate"].items():
        print(f"  {n_:<5}{v['mean_abs']:>14.4e}{v['share_pct']:>9.1f}%"
              f"{v['sensitivity_per_unit']:>18.4e}{v['share_pct_normalized']:>11.1f}%")
    print("    -> raw 비중은 기준점까지의 거리에 비례하므로, 물리적 중요도는 "
          "정규화 비중으로 읽어야 함")
    print(f"  완비성 공리 잔차 = {xai['completeness_gap']:.3e} (0 에 가까울수록 정확)")
    print(f"  X-Y 비대칭도 = {xai['xy_asymmetry']:.4f} "
          f"(소스항이 x,y 대칭이므로 0 에 가까워야 물리적으로 타당)")

    print("\nStep 8: FDM 대비 속도 실측 ...")
    geo0 = create_geological_model(cfg, pc.phi_mean_mu, pc.eps_mu,
                                   rng=np.random.default_rng(1000))
    bm = benchmark(cfg, model, geo0, device)
    print(f"  FDM 전체 해석 {bm['fdm_full_solve_s']:.2f} s  |  "
          f"PINN {bm['pinn_points']:,}점 추론 {bm['pinn_infer_s']*1e3:.2f} ms")

    print("\nStep 9: 그림 및 결과 저장 ...")
    make_figures(cfg, geo0, pce, blind, xai, rl, history)
    summary = {
        "config": {"reservoir": asdict(cfg), "pipeline": asdict(pc)},
        "blind": {k: v for k, v in blind.items() if k not in ("truth", "pred")},
        "surrogate_bias": bias,
        "pce_at_30yr": {
            "mean_mpa": float(cfg.to_mpa(pce["mean"][j30])),
            "std_mpa": float(pce["std"][j30] * cfg.dp_ref / MPA),
            "p975_mpa": float(cfg.to_mpa(pce["p975"][j30])),
            "sobol_first": pce["sobol_first"][j30].tolist(),
            "sobol_total": pce["sobol_total"][j30].tolist(),
        },
        "rl": rl,
        "xai": {"aggregate": xai["aggregate"], "completeness_gap": xai["completeness_gap"],
                "xy_asymmetry": xai["xy_asymmetry"]},
        "benchmark": bm,
        "table1": {
            "porosity": basic_stats(geo0.porosity),
            "permeability_mD": basic_stats(geo0.permeability_mD),
            "normalized_log_permeability": basic_stats(geo0.perm_norm),
            "seismic": basic_stats(geo0.seismic_time),
            "well_log_zscore": basic_stats(geo0.well_log_z),
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"  -> {OUT/'summary.json'} , fig1~fig5 저장 완료")


if __name__ == "__main__":
    main()
