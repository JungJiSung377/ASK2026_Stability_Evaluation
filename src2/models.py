"""
models.py — 파라미터화 물리정보 신경망 · 물리 손실 · 설명가능성(IG)

원본 코드 대비 수정 내역 (감사 태그)
------------------------------------
[C2]  입력을 (x,y,z,t) -> (x,y,z,t,phi,eps) 6차원으로 확장한 '파라미터화 PINN'.
      원본은 지질 파라미터가 입력이 아니어서 PCE 가 PINN 이 아닌 손수 쓴 수식을
      대리모델링했다. 이제 표본마다 서로 다른 압력장을 신경망이 직접 내놓는다.
[C3]  초기조건을 하드 제약으로 내장(p(t=0) = p_init 이 구조적으로 보장)하고,
      경계조건 손실과 관측 데이터 손실을 별도 항으로 추가.
[C5]  안전 제약(p <= 45MPa)을 학습 손실에서 제거. 물리 대리모델을 안전 기준에
      맞춰 훈련하면 위험을 과소평가하도록 편향되고 검증이 순환에 빠진다.
      안전은 '진단 지표'로만 계산해 강화학습 보상에서 부과한다.
[M2]  div(k grad p) = k Lap(p) + grad(k).grad(p) 의 두 번째 항을 명시적으로 포함.
      원본은 k 가 상수 텐서라 autograd 가 이 항을 생성하지 않았다(누락분 RMS 65%).
[M3]  좌표를 [-1,1] 로 정규화하고 무차원 확산계수 A_x/A_y/A_z 를 명시적으로 사용.
      FDM(h = 2/n)과 동일한 **셀 중심** 규약을 쓴다.
[m4]  표준 Xavier gain(tanh: 5/3) 사용. 원본의 gain=0.1 은 출력 억제 트릭이었다.
[m5]  coords 를 함수 내부에서 clone 해 리프 텐서에 grad 가 누적되지 않도록 함.
[m13] Integrated Gradients 를 중점법 리만합으로 계산하고 완비성 공리를 검산.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from environment import (
    SEC_PER_YEAR,
    ReservoirConfig,
    injection_schedule,
    source_shape_grid,
)


# ---------------------------------------------------------------------------
# [1] 이질 물성장의 미분가능 보간 — [M2] grad(k) 를 정확히 공급하기 위한 장치
# ---------------------------------------------------------------------------
class HeterogeneousField:
    """여러 지질 실현의 k* 와 그 공간 도함수를 담고 삼선형 보간으로 질의한다.

    k 는 신경망 파라미터에 의존하지 않으므로 상수로 취급해도 되지만,
    잔차에 grad(k).grad(p) 항이 반드시 들어가야 하므로 k 의 공간 도함수를
    미리 계산해 함께 보간한다. (원본은 이 항이 통째로 빠져 있었다.)
    """

    def __init__(self, cfg: ReservoirConfig, k_star_stack: np.ndarray, device):
        # k_star_stack: (n_realizations, nx, ny, nz)
        self.cfg = cfg
        self.shape = k_star_stack.shape[1:]

        # 정규화 좌표 xi in [-1,1] 에 대한 도함수: d/dxi = (n/2) * d/d(index)
        gx = np.gradient(k_star_stack, axis=1) * (cfg.nx / 2.0)
        gy = np.gradient(k_star_stack, axis=2) * (cfg.ny / 2.0)
        gz = np.gradient(k_star_stack, axis=3) * (cfg.nz / 2.0)

        to = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
        self.k = to(k_star_stack)
        self.gk = torch.stack([to(gx), to(gy), to(gz)], dim=0)  # (3, R, nx, ny, nz)

    def _weights(self, xi, n):
        """정규화 좌표 -> (하단 인덱스, 상단 인덱스, 상단 가중치).

        **셀 중심** 규약: xi = 2*(i + 0.5)/n - 1.  FDM(h = 2/n)과 Normalizer 가
        모두 같은 규약을 쓰지 않으면 grad(k) 스케일과 소스항 총량이 어긋난다.
        """
        pos = (xi + 1.0) * 0.5 * n - 0.5
        pos = pos.clamp(0.0, n - 1 - 1e-6)
        i0 = pos.floor().long()
        return i0, (i0 + 1).clamp(max=n - 1), pos - i0.to(pos.dtype)

    def sample(self, r_idx, xi, eta, zeta):
        """삼선형 보간으로 (k, dk/dxi, dk/deta, dk/dzeta) 를 반환.

        모두 detach 한다. 보간 가중치가 좌표에 미분가능하게 의존하므로, detach 하지
        않으면 autograd 가 보간을 통해서도 미분해 grad(k) 항이 이중 계상된다.
        여기서는 grad(k) 를 명시적으로 따로 공급하므로 k 는 상수여야 한다.
        """
        nx, ny, nz = self.shape
        i0, i1, wx = self._weights(xi, nx)
        j0, j1, wy = self._weights(eta, ny)
        k0, k1, wz = self._weights(zeta, nz)

        def gather(vol):
            out = 0.0
            for ii, cx in ((i0, 1 - wx), (i1, wx)):
                for jj, cy in ((j0, 1 - wy), (j1, wy)):
                    for kk, cz in ((k0, 1 - wz), (k1, wz)):
                        out = out + cx * cy * cz * vol[r_idx, ii, jj, kk]
            return out

        k = gather(self.k)
        grads = [gather(self.gk[d]) for d in range(3)]
        return k.detach(), [g.detach() for g in grads]


# ---------------------------------------------------------------------------
# [2] 파라미터화 PINN
# ---------------------------------------------------------------------------
class PressurePINN(nn.Module):
    """무차원 과압 p*(xi, eta, zeta, tau; phi, eps) 를 근사한다.

    출력 규약
    ---------
        p*  = (1 - exp(-t_yr / tau_ic)) * net(...)

    이 형태는 t = 0 에서 항등적으로 0 을 주므로 초기조건
    p(t=0) = p_init 이 **하드 제약**으로 만족된다. 원본은 초기조건 손실도
    없었고 +30.0 오프셋과 gain=0.1 초기화가 우연히 그 역할을 대신했다.
    """

    def __init__(self, cfg: ReservoirConfig, width: int = 128, depth: int = 5,
                 tau_ic: float = 0.15):
        super().__init__()
        self.cfg = cfg
        self.tau_ic = tau_ic

        layers, d_in = [], 6
        for i in range(depth - 1):
            d_out = width if i < depth - 2 else width // 2
            layers += [nn.Linear(d_in, d_out), nn.Tanh()]
            d_in = d_out
        layers += [nn.Linear(d_in, 1)]
        self.net = nn.Sequential(*layers)

        # [m4] tanh 에 대한 표준 Xavier gain
        gain = nn.init.calculate_gain("tanh")
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def forward(self, xi, eta, zeta, tau, phi_n, eps_n):
        """모든 입력은 이미 [-1,1] 근방으로 정규화되어 있다고 가정한다.

        tau 는 '연 단위 시간 / horizon_years' 로 정규화된 값이며,
        하드 초기조건에 쓰기 위해 물리 연도로 되돌려 사용한다.
        """
        raw = self.net(torch.cat([xi, eta, zeta, tau, phi_n, eps_n], dim=1))
        # 물리적으로 t < 0 은 존재하지 않는다. clamp 를 빼면 음수 tau 가 들어왔을 때
        # exp(+큰수) 가 float32 에서 overflow 하여 출력이 -inf/NaN 이 된다.
        t_years = (tau * self.cfg.horizon_years).clamp_min(0.0)
        ramp = 1.0 - torch.exp(-t_years / self.tau_ic)
        return ramp * raw

    def pressure_mpa(self, xi, eta, zeta, tau, phi_n, eps_n):
        """진단용: 절대압력 [MPa]. 학습 손실에는 쓰지 않는다."""
        return self.cfg.to_mpa(self.forward(xi, eta, zeta, tau, phi_n, eps_n))


# ---------------------------------------------------------------------------
# [3] 정규화 헬퍼 — 물리 좌표 <-> 신경망 입력
# ---------------------------------------------------------------------------
class Normalizer:
    """[M3] 좌표·파라미터를 [-1,1] 로 사상. 원본은 [0,63] 원시값을 tanh 에 직접 넣었다."""

    def __init__(self, cfg: ReservoirConfig, phi_range, eps_range):
        self.cfg = cfg
        self.phi_lo, self.phi_hi = phi_range
        self.eps_lo, self.eps_hi = eps_range

    @staticmethod
    def _to_unit(v, lo, hi):
        return 2.0 * (v - lo) / (hi - lo) - 1.0

    def index_to_xi(self, ix, iy, iz):
        """셀 중심 규약:  xi = 2*(i + 0.5)/n - 1  (FDM 의 h = 2/n 과 일치)."""
        c = self.cfg
        return (self._to_unit(ix, -0.5, c.nx - 0.5),
                self._to_unit(iy, -0.5, c.ny - 0.5),
                self._to_unit(iz, -0.5, c.nz - 0.5))

    def xi_to_index(self, xi, eta, zeta):
        c = self.cfg
        return ((xi + 1.0) * 0.5 * c.nx - 0.5,
                (eta + 1.0) * 0.5 * c.ny - 0.5,
                (zeta + 1.0) * 0.5 * c.nz - 0.5)

    def year_to_tau(self, t_years):
        return t_years / self.cfg.horizon_years

    def params_to_unit(self, phi, eps):
        return (self._to_unit(phi, self.phi_lo, self.phi_hi),
                self._to_unit(eps, self.eps_lo, self.eps_hi))


# ---------------------------------------------------------------------------
# [4] 물리 손실 — [M2][M3][C3][C5]
# ---------------------------------------------------------------------------
def source_term(cfg: ReservoirConfig, ix, iy, iz, t_years, phi_mean, norm_const: float):
    """PINN 잔차용 소스항. FDM 기준해와 **동일한 형상함수/스케줄**을 사용한다."""
    shape = source_shape_grid(cfg, ix, iy, iz) / norm_const
    strength = cfg.source_strength(1.0) / phi_mean          # source_strength 는 1/phi 비례
    return strength * shape * injection_schedule(cfg, t_years)


def compute_pde_residual(
    model: PressurePINN,
    normalizer: Normalizer,
    field: HeterogeneousField,
    r_idx: torch.Tensor,
    coords: torch.Tensor,       # (N, 4): xi, eta, zeta, tau
    phi: torch.Tensor,          # (N, 1) 물리 공극률 평균
    eps: torch.Tensor,          # (N, 1) KC 잔차 log10
    norm_const: float,
) -> torch.Tensor:
    """무차원 잔차

        R = dp/dt_yr - sum_i A_i [ k d2p/dxi_i2 + (dk/dxi_i)(dp/dxi_i) ] - s

    원본과의 결정적 차이는 두 번째 항 (dk/dxi)(dp/dxi) 이 실제로 들어간다는 점이다.
    """
    cfg = model.cfg
    # [m5] 리프 텐서에 grad 가 누적되지 않도록 내부에서 clone
    c = coords.clone().requires_grad_(True)
    phi_n, eps_n = normalizer.params_to_unit(phi, eps)

    p = model(c[:, [0]], c[:, [1]], c[:, [2]], c[:, [3]], phi_n, eps_n)

    g1 = torch.autograd.grad(p, c, torch.ones_like(p), create_graph=True)[0]
    dp = [g1[:, [i]] for i in range(3)]
    dp_dtau = g1[:, [3]]

    d2p = []
    for i in range(3):
        gi = torch.autograd.grad(dp[i], c, torch.ones_like(dp[i]), create_graph=True)[0]
        d2p.append(gi[:, [i]])

    # 이질 투과율과 그 공간 도함수(상수)
    k, gk = field.sample(r_idx, c[:, 0], c[:, 1], c[:, 2])
    k = k.unsqueeze(1)
    gk = [g.unsqueeze(1) for g in gk]

    # 표본별 무차원 확산계수 A_i(phi)
    hx, hy, hz = cfg.half_extent
    scale = cfg.eta_ref * SEC_PER_YEAR * (cfg.phi_ref / phi)
    a = [scale / hx**2, scale / hy**2, scale / hz**2 * cfg.kv_kh]

    divergence = sum(a[i] * (k * d2p[i] + gk[i] * dp[i]) for i in range(3))

    # 시간 미분: tau 는 horizon 으로 정규화되어 있으므로 연 단위로 환산
    dp_dyear = dp_dtau / cfg.horizon_years

    ix, iy, iz = normalizer.xi_to_index(c[:, [0]], c[:, [1]], c[:, [2]])
    t_years = c[:, [3]] * cfg.horizon_years
    s = source_term(cfg, ix, iy, iz, t_years, phi, norm_const)

    return dp_dyear - divergence - s


def compute_boundary_residual(
    model: PressurePINN,
    normalizer: Normalizer,
    coords: torch.Tensor,
    phi: torch.Tensor,
    eps: torch.Tensor,
    face_axis: torch.Tensor,     # (N,) 0/1/2 어느 축의 면인지 (int64)
    is_lateral: torch.Tensor,    # (N,) bool — 측면(Dirichlet) 여부
) -> torch.Tensor:
    """[C3] 경계조건 잔차.

    측면(x, y): 원거리 정압 -> p* = 0
    상/하부(z): 덮개암·하부 차수층 무유동 -> dp/dzeta = 0
    """
    c = coords.clone().requires_grad_(True)
    phi_n, eps_n = normalizer.params_to_unit(phi, eps)
    p = model(c[:, [0]], c[:, [1]], c[:, [2]], c[:, [3]], phi_n, eps_n)

    g = torch.autograd.grad(p, c, torch.ones_like(p), create_graph=True)[0]
    normal_grad = torch.gather(g[:, :3], 1, face_axis.view(-1, 1))

    lateral = is_lateral.view(-1, 1).to(p.dtype)
    return lateral * p + (1.0 - lateral) * normal_grad


def compute_data_residual(
    model: PressurePINN,
    normalizer: Normalizer,
    coords: torch.Tensor,
    phi: torch.Tensor,
    eps: torch.Tensor,
    p_observed: torch.Tensor,
) -> torch.Tensor:
    """[C3] 관측 데이터 손실. 논문 식(1)의 Loss_data 에 해당하며 원본에는 없었다.

    관측치는 FDM 기준해에서 뽑은 '희소한 모니터링 정' 값이다.
    """
    phi_n, eps_n = normalizer.params_to_unit(phi, eps)
    p = model(coords[:, [0]], coords[:, [1]], coords[:, [2]], coords[:, [3]], phi_n, eps_n)
    return p - p_observed


def safety_diagnostic(p_star: torch.Tensor, cfg: ReservoirConfig) -> torch.Tensor:
    """[C5] 임계압력 초과량 — **진단 전용**.

    원본은 이 값을 학습 손실에 더해 신경망이 45MPa 를 넘지 않도록 훈련시켰고,
    그 모델로 다시 45MPa 초과 위험을 평가해 순환에 빠졌다. 여기서는 손실이
    아니라 사후 지표로만 쓰며, 실제 제약은 강화학습 보상에서 부과한다.

    [M8] 명칭에 관한 주의: 이것은 압력 단일 임계값 기준이지 Mohr-Coulomb 파괴
    기준이 아니다. Mohr-Coulomb 은 tau = c + (sigma_n - alpha*p) tan(phi_f) 로
    응력텐서·점착력·내부마찰각·Biot 계수를 요구한다. 45 MPa 는 파쇄압의 보수적
    대용치이므로, 논문에서도 'Mohr-Coulomb 적용'이 아니라 '임계압력 제약'으로
    서술해야 한다.
    """
    limit_star = (cfg.p_limit - cfg.p_init) / cfg.dp_ref
    return torch.relu(p_star - limit_star)


# ---------------------------------------------------------------------------
# [5] Integrated Gradients — [m13] 중점법 + 완비성 검산
# ---------------------------------------------------------------------------
def compute_integrated_gradients(
    model: PressurePINN,
    inputs: torch.Tensor,        # (N, 6) 정규화된 입력
    baseline: torch.Tensor,      # (N, 6) 또는 (1, 6)
    steps: int = 64,
):
    """IG_i = (x_i - x'_i) * integral_0^1 dF/dx_i (x' + a (x - x')) da

    반환: (attributions, completeness_gap)
      completeness_gap = | sum_i IG_i - (F(x) - F(x')) | 의 평균.
      원본은 linspace(0,1,steps) 로 양 끝점을 모두 포함하면서 1/steps 로 나눠
      완비성 공리가 근사적으로만 성립했다. 여기서는 중점법을 쓴다.
    """
    model.eval()
    baseline = baseline.expand_as(inputs)
    delta = inputs - baseline
    total = torch.zeros_like(inputs)

    # [m13] 중점법: alpha_k = (k + 0.5) / steps
    alphas = (torch.arange(steps, device=inputs.device, dtype=inputs.dtype) + 0.5) / steps
    for alpha in alphas:
        pt = (baseline + alpha * delta).clone().detach().requires_grad_(True)
        out = model(*[pt[:, [i]] for i in range(6)])
        grad = torch.autograd.grad(out, pt, torch.ones_like(out))[0]
        total = total + grad / steps

    attributions = delta * total

    with torch.no_grad():
        f_x = model(*[inputs[:, [i]] for i in range(6)])
        f_b = model(*[baseline[:, [i]] for i in range(6)])
    gap = (attributions.sum(dim=1, keepdim=True) - (f_x - f_b)).abs().mean()
    return attributions, float(gap)


def aggregate_attributions(attributions: torch.Tensor, delta: torch.Tensor | None = None,
                           names=None) -> dict:
    """[M13] 원본은 attributions 를 계산만 하고 한 번도 집계·출력하지 않았다.

    주의: IG 기여도는 정의상 (x_i - x'_i) 에 비례하므로, 기준점에서 멀리 떨어진
    입력일수록 크게 나온다. 이는 물리적 중요도가 아니라 기준점까지의 거리다.
    따라서 raw 기여도와 함께 |delta| 로 정규화한 '단위 입력당 민감도'를 보고한다.
    """
    names = names or ["x", "y", "z", "t", "phi", "eps"]
    mean_abs = attributions.abs().mean(dim=0).detach().cpu().numpy()
    total = float(mean_abs.sum()) + 1e-30
    span = per_unit = None
    if delta is not None:
        span = delta.abs().mean(dim=0).detach().cpu().numpy()
        per_unit = mean_abs / np.maximum(span, 1e-12)
        per_unit_total = float(per_unit.sum()) + 1e-30
    out = {}
    for i, (n, v) in enumerate(zip(names, mean_abs)):
        rec = {"mean_abs": float(v), "share_pct": float(100.0 * v / total)}
        if delta is not None:
            rec["mean_abs_delta"] = float(span[i])
            rec["sensitivity_per_unit"] = float(per_unit[i])
            rec["share_pct_normalized"] = float(100.0 * per_unit[i] / per_unit_total)
        out[n] = rec
    return out
