"""
verify_fixes.py — 수정본의 물리·수치 정합성 검증 (torch / chaospy / SB3 불필요)

numpy + scipy 만으로 돌아가며, 아래 항목을 정량 확인한다.

  A. 주입 스케줄의 해석적 시간 평균이 중점 표본보다 정확한가
  B. FDM 기준해가 시간 스텝 세분에 대해 수렴하는가          <- 기준해로 쓸 자격
  C. 좌표 규약(셀 중심)이 FDM 과 PINN 사이에서 일치하는가    <- 소스 총량 정합
  D. 같은 (phi, eps) 가 항상 같은 k* 를 주는가              <- PINN 학습 대상의 결정성
  E. 안전 제약이 최대 주입에서 실제로 도달 가능한가          <- 가중치 스캔의 유효성
  F. div(k grad p) 전개형 공식이 정확한가 (제조해, O(h^2) 수렴)
  G. grad(k).grad(p) 누락 시 오차 크기 (원본 M2 정량화)
  H. 콜로케이션 중요도 표집이 소스 영역을 실제로 덮는가

실행:  python verify_fixes.py                    (전체, 수 분~수십 분)
       python verify_fixes.py --fast             (격자·스텝 축소)
       python verify_fixes.py --quick            (--fast 와 동일)

main.py 와의 관계
-----------------
  main.py         : 전체 파이프라인 실행 (torch / chaospy / stable-baselines3 필요)
  verify_fixes.py : 물리·수치 정합성만 점검 (numpy / scipy 만 필요)
두 스크립트 모두 --quick 과 --fast 를 같은 의미로 받는다.
"""

from __future__ import annotations

import argparse
import sys
import time
import types

import numpy as np

# ---------------------------------------------------------------------------
# gymnasium 이 없어도 environment.py 를 임포트할 수 있도록 최소 스텁을 심는다.
# (검증 대상은 numpy/scipy 경로이며, 강화학습 환경 클래스는 정의만 되면 된다.)
# ---------------------------------------------------------------------------
if "gymnasium" not in sys.modules:
    try:
        import gymnasium  # noqa: F401
    except ImportError:
        _gym = types.ModuleType("gymnasium")
        _spaces = types.ModuleType("gymnasium.spaces")

        class _Box:
            def __init__(self, low, high, shape=None, dtype=np.float32):
                shp = shape if shape is not None else np.shape(low)
                self.low = np.broadcast_to(np.asarray(low, dtype=dtype), shp).astype(dtype)
                self.high = np.broadcast_to(np.asarray(high, dtype=dtype), shp).astype(dtype)
                self.shape = self.low.shape
                self.dtype = dtype

        class _Env:
            def reset(self, seed=None, options=None):
                self.np_random = np.random.default_rng(seed)

        _spaces.Box = _Box
        _gym.Env = _Env
        _gym.spaces = _spaces
        sys.modules["gymnasium"] = _gym
        sys.modules["gymnasium.spaces"] = _spaces

from environment import (  # noqa: E402
    FDMReference,
    ReservoirConfig,
    create_geological_model,
    injection_schedule,
    injection_schedule_mean,
    source_shape_grid,
)


def rule(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_normalizer(cfg):
    """models.py 의 Normalizer 만 torch 없이 떼어내 실행한다."""
    src = open("models.py", encoding="utf-8").read()
    start = src.index("class Normalizer")
    end = src.index("def source_term(")
    ns = {"np": np, "ReservoirConfig": ReservoirConfig}
    exec(compile(src[start:end], "<models.Normalizer>", "exec"), ns)  # noqa: S102
    return ns["Normalizer"](cfg, (0.129, 0.249), (-1.2, 1.2))


def main() -> int:
    ap = argparse.ArgumentParser()
    # main.py 는 --quick, 여기는 --fast 였다. 헷갈리므로 둘 다 받는다.
    ap.add_argument("--fast", "--quick", dest="fast", action="store_true",
                    help="격자·스텝을 줄여 빠르게 점검")
    args = ap.parse_args()

    cfg = ReservoirConfig()
    failures = []

    # ---------------------------------------------------------------- A
    rule("A) 주입 스케줄: 중점 표본 vs 해석적 시간 평균")
    worst = 0.0
    for t0, t1 in [(28.0, 30.0), (29.0, 30.0), (30.0, 32.0), (26.97, 33.56)]:
        mid = float(injection_schedule(cfg, 0.5 * (t0 + t1)))
        exact = injection_schedule_mean(cfg, t0, t1)
        worst = max(worst, abs(mid - exact))
        print(f"  [{t0:6.2f}, {t1:6.2f}]  중점 {mid:.4f}   해석적 평균 {exact:.4f}"
              f"   차이 {abs(mid - exact):.4f}")
    print(f"  -> 스텝이 종료 전이폭(0.25 yr)보다 크면 중점 표본은 주입량을 왜곡한다"
          f" (최대 차이 {worst:.3f})")

    # ---------------------------------------------------------------- B
    rule("B) FDM 기준해의 시간 수렴성  [기준해로 쓸 자격]")
    geo = create_geological_model(cfg, 0.189, 0.0, rng=np.random.default_rng(1000))
    fdm = FDMReference(cfg, geo)
    probe = np.array([0.0, cfg.inj_years])
    scales = (2.0, 1.0, 0.5) if args.fast else (2.0, 1.0, 0.5, 0.25)
    peaks = []
    for s in scales:
        t0 = time.time()
        sol = fdm.solve(probe, dt_scale=s)
        peak = float(cfg.to_mpa(sol[-1].max()))
        peaks.append(peak)
        delta = "" if len(peaks) < 2 else f"   직전 대비 {abs(peak - peaks[-2]):+.4f} MPa"
        print(f"  dt_scale={s:<5} 스텝 {len(fdm.build_time_grid(cfg.inj_years, s)):>4}개"
              f"  30년 최대압 {peak:8.4f} MPa  ({time.time() - t0:5.0f}s){delta}")
    drift = abs(peaks[-1] - peaks[-2])
    ok = drift < 0.05 * max(abs(peaks[-1] - 30.0), 1e-9)
    print(f"  -> 최종 두 설정 간 변화 {drift:.4f} MPa  "
          f"({'수렴 OK' if ok else '미수렴 — build_time_grid 를 더 세분할 것'})")
    if not ok:
        failures.append("B: FDM 시간 수렴 미달")

    # ---------------------------------------------------------------- C
    rule("C) 좌표 규약 일치 (셀 중심) — PINN 소스 적분 vs FDM 총 주입량")
    nz_ = load_normalizer(cfg)
    a = np.array([0.0, 17.0, 63.0])
    b = np.array([0.0, 31.0, 63.0])
    c = np.array([0.0, 7.0, 31.0])
    xi, eta, ze = nz_.index_to_xi(a, b, c)
    ra, rb, rc = nz_.xi_to_index(xi, eta, ze)
    rt = max(np.abs(ra - a).max(), np.abs(rb - b).max(), np.abs(rc - c).max())
    print(f"  index -> xi -> index 왕복 최대 오차 = {rt:.2e}")
    xi0 = nz_.index_to_xi(np.array([0.0]), np.array([0.0]), np.array([0.0]))[0][0]
    print(f"  셀 중심 규약: index 0 -> xi {xi0:+.6f}   (기대 {2 * 0.5 / cfg.nx - 1:+.6f})")

    # 소스는 도메인의 ~0.35 % 에만 집중되어 있어 균일 몬테카를로로는 표본이 거의
    # 들어가지 않는다. 형상함수가 radial(i,j) * window(k) 로 분리되므로
    # 각 인자를 결정론적 구적으로 적분한다.
    sub = 8                                   # 셀당 세분 수
    z_deep = cfg.nz * 0.72                    # window == 1 인 깊이 (radial 인자 분리용)
    x_mid, y_mid = cfg.nx / 2.0 - 0.5, cfg.ny / 2.0 - 0.5

    def fine_axis(n):
        return np.arange(n * sub, dtype=float) / sub - 0.5 + 0.5 / sub

    def radial(aa, bb):
        A, B = np.meshgrid(aa, bb, indexing="ij")
        return source_shape_grid(cfg, A, B, np.full(A.shape, z_deep))

    def window(k):
        return source_shape_grid(cfg, np.full(k.shape, x_mid), np.full(k.shape, y_mid), k)

    r_ratio = float(radial(fine_axis(cfg.nx), fine_axis(cfg.ny)).sum() / sub**2
                    / radial(np.arange(cfg.nx, dtype=float),
                             np.arange(cfg.ny, dtype=float)).sum())
    w_ratio = float(window(fine_axis(cfg.nz)).sum() / sub
                    / window(np.arange(cfg.nz, dtype=float)).sum())
    integral = r_ratio * w_ratio
    print(f"  수평 인자 적분/합 = {r_ratio:.6f}   수직 인자 적분/합 = {w_ratio:.6f}")
    print(f"  PINN 소스 적분 / FDM 총 주입량 = {integral:.4f}   (1.0 이어야 함)")
    if abs(integral - 1.0) > 0.02:
        failures.append(f"C: 소스 총량 불일치 {integral:.4f}")

    # ---------------------------------------------------------------- D
    rule("D) 동일 (phi, eps) -> 동일 k*  [PINN 학습 대상의 결정성]")
    g1 = create_geological_model(cfg, 0.189, 0.0, rng=np.random.default_rng(1000))
    g2 = create_geological_model(cfg, 0.189, 0.0, rng=np.random.default_rng(1000))
    dk = float(np.abs(g1.k_star - g2.k_star).max())
    print(f"  같은 시드 두 번 호출 시 k* 최대 차이 = {dk:.2e}   (0 이어야 함)")
    if dk > 0:
        failures.append("D: k* 가 결정적이지 않음")

    # ---------------------------------------------------------------- E
    rule("E) 안전 제약 도달 가능성  [가중치 스캔이 의미를 가지려면 필수]")
    iz_mon = int(round(0.72 * (cfg.nz - 1)))
    designs = [(0.189, 0.0), (0.189, -0.44), (0.165, -0.30), (0.213, 0.30)]
    print(f"  {'phi':>7}{'eps':>7}{'개공중앙':>12}{'덮개암상단':>12}{'도메인최대':>12}"
          f"{'a=1.5 환산':>12}")
    reachable = False
    for phim, eps in designs:
        gg = create_geological_model(cfg, phim, eps, rng=np.random.default_rng(1000))
        sol = FDMReference(cfg, gg).solve(np.array([0.0, cfg.inj_years]))[-1]
        mon = float(cfg.to_mpa(sol[cfg.nx // 2, cfg.ny // 2, iz_mon]))
        cap = float(cfg.to_mpa(sol[cfg.nx // 2, cfg.ny // 2, 0]))
        dom = float(cfg.to_mpa(sol.max()))
        amp = 30.0 + (mon - 30.0) * 1.5
        reachable |= amp > cfg.p_limit / 1e6
        print(f"  {phim:>7.3f}{eps:>7.2f}{mon:>12.2f}{cap:>12.2f}{dom:>12.2f}{amp:>12.2f}")
    print(f"  임계압력 = {cfg.p_limit / 1e6:.0f} MPa")
    print(f"  -> 최대 주입에서 임계 초과 "
          f"{'발생 (제약 활성)' if reachable else '없음 (제약 비활성)'}")
    if not reachable:
        failures.append("E: 안전 제약이 비활성 — inject_mt_per_year 를 올릴 것")

    # ---------------------------------------------------------------- F
    rule("F) 제조해 검증: div(k grad p) = k Lap(p) + grad(k).grad(p)")
    A = list(cfg.diffusion_coeffs(0.189))

    def k_fn(x, y, z):
        return 1.0 + 0.4 * np.sin(1.7 * x) * np.cos(1.3 * y) + 0.25 * np.sin(0.9 * z)

    def dk_fn(x, y, z):
        return [0.4 * 1.7 * np.cos(1.7 * x) * np.cos(1.3 * y),
                -0.4 * 1.3 * np.sin(1.7 * x) * np.sin(1.3 * y),
                0.25 * 0.9 * np.cos(0.9 * z)]

    def p_fn(x, y, z):
        return np.sin(1.1 * x) * np.cos(0.8 * y) * np.sin(1.4 * z)

    def dp_fn(x, y, z):
        return [1.1 * np.cos(1.1 * x) * np.cos(0.8 * y) * np.sin(1.4 * z),
                -0.8 * np.sin(1.1 * x) * np.sin(0.8 * y) * np.sin(1.4 * z),
                1.4 * np.sin(1.1 * x) * np.cos(0.8 * y) * np.cos(1.4 * z)]

    def exact(x, y, z):
        k, dk_, dp_, p_ = k_fn(x, y, z), dk_fn(x, y, z), dp_fn(x, y, z), p_fn(x, y, z)
        d2 = [-1.1**2 * p_, -0.8**2 * p_, -1.4**2 * p_]
        return sum(A[i] * (k * d2[i] + dk_[i] * dp_[i]) for i in range(3))

    prev = None
    grids = (24, 48, 96) if args.fast else (24, 48, 96, 192)
    print(f"  {'격자 n':>8}{'h':>10}{'보존형 FD 대비 상대오차':>26}{'수렴차수':>10}")
    for n in grids:
        ax = np.linspace(-1, 1, n)
        h = ax[1] - ax[0]
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        K, P = k_fn(X, Y, Z), p_fn(X, Y, Z)
        cons = np.zeros_like(P)
        for axis, coef in enumerate(A):
            s1 = [slice(None)] * 3
            s2 = [slice(None)] * 3
            s1[axis] = slice(0, -1)
            s2[axis] = slice(1, None)
            kl, kr = K[tuple(s1)], K[tuple(s2)]
            flux = coef * (2 * kl * kr / (kl + kr)) * (P[tuple(s2)] - P[tuple(s1)]) / h**2
            cons[tuple(s1)] += flux
            cons[tuple(s2)] -= flux
        ref = exact(X, Y, Z)
        itr = (slice(2, -2),) * 3
        err = float(np.sqrt(np.mean((cons[itr] - ref[itr]) ** 2))
                    / np.sqrt(np.mean(ref[itr] ** 2)))
        order = "" if prev is None else f"{np.log2(prev / err):.2f}"
        print(f"  {n:>8}{h:>10.4f}{err * 100:>24.4f} %{order:>10}")
        prev = err
    print("  -> 오차가 h^2 로 수렴하면 전개형 공식과 A_i 계수가 정확하다는 뜻")

    # ---------------------------------------------------------------- G
    rule("G) 실제 k 필드에서 grad(k).grad(p) 항의 크기  [원본 M2 정량화]")
    sol = fdm.solve(np.array([5.0]))[0].astype(np.float64)
    k = geo.k_star.astype(np.float64)
    n = [cfg.nx, cfg.ny, cfg.nz]

    def d1(arr, axis):
        h = 2.0 / n[axis]
        out = np.zeros_like(arr)
        sl = [slice(None)] * 3; sl[axis] = slice(1, -1)
        pl = [slice(None)] * 3; pl[axis] = slice(2, None)
        mn = [slice(None)] * 3; mn[axis] = slice(0, -2)
        out[tuple(sl)] = (arr[tuple(pl)] - arr[tuple(mn)]) / (2 * h)
        return out

    def d2(arr, axis):
        h = 2.0 / n[axis]
        out = np.zeros_like(arr)
        sl = [slice(None)] * 3; sl[axis] = slice(1, -1)
        pl = [slice(None)] * 3; pl[axis] = slice(2, None)
        mn = [slice(None)] * 3; mn[axis] = slice(0, -2)
        out[tuple(sl)] = (arr[tuple(pl)] - 2 * arr[tuple(sl)] + arr[tuple(mn)]) / h**2
        return out

    keep = sum(A[i] * k * d2(sol, i) for i in range(3))
    miss = sum(A[i] * d1(k, i) * d1(sol, i) for i in range(3))
    itr = (slice(3, -3),) * 3
    rk = float(np.sqrt(np.mean(keep[itr] ** 2)))
    rm = float(np.sqrt(np.mean(miss[itr] ** 2)))
    print(f"  RMS |k Lap(p)|        = {rk:.4e}   (원본이 계산하던 항)")
    print(f"  RMS |grad(k).grad(p)| = {rm:.4e}   (원본이 누락한 항)")
    print(f"  누락항 / 유지항 = {rm / rk * 100:.1f} %")

    # ---------------------------------------------------------------- H
    rule("H) 콜로케이션 중요도 표집이 소스 영역을 덮는가")
    rng = np.random.default_rng(0)
    sig_r, n_batch = 1.2, 4096

    def sample(n_pts, frac):
        n_well = int(round(n_pts * frac))
        n_uni = n_pts - n_well
        parts = []
        if n_uni:
            parts.append(rng.uniform(-1, 1, (n_uni, 3)))
        if n_well:
            r_min, r_max = 1.0 / cfg.nx, 0.6
            r = r_min * (r_max / r_min) ** rng.random(n_well)
            th = rng.random(n_well) * 2 * np.pi
            parts.append(np.stack([
                np.clip(r * np.cos(th), -1, 1),
                np.clip(r * np.sin(th), -1, 1),
                np.clip(rng.random(n_well) * 1.3 - 0.3, -1, 1)], 1))
        return np.concatenate(parts)

    def in_source(p):
        ix = (p[:, 0] + 1) * 0.5 * cfg.nx - 0.5
        iy = (p[:, 1] + 1) * 0.5 * cfg.ny - 0.5
        iz = (p[:, 2] + 1) * 0.5 * cfg.nz - 0.5
        r = np.hypot(ix - (cfg.nx / 2 - 0.5), iy - (cfg.ny / 2 - 0.5))
        return (r <= 3 * sig_r) & (iz >= cfg.nz * 0.55) & (iz <= cfg.nz * 0.90)

    reps = 50 if args.fast else 200
    for frac, label in [(0.0, "균일만 (원본 방식)"), (0.5, "균일 50% + 근정 50%")]:
        hits = [in_source(sample(n_batch, frac)).sum() for _ in range(reps)]
        print(f"  {label:<24} 배치 {n_batch:,}점 중 소스영역 표본 "
              f"= 평균 {np.mean(hits):7.1f}개")
    print("  -> 균일 표집만으로는 '어디서나 0' 이 손실 최소해가 되어 학습이 실패한다")

    # ---------------------------------------------------------------- 요약
    rule("검증 요약")
    if failures:
        for f in failures:
            print(f"  [실패] {f}")
        return 1
    print("  모든 항목 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
