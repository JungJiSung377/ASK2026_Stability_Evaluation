"""
environment.py — 합성 지질 모델 · 유한차분 기준해 솔버 · 강화학습 주입 제어 환경

CO2 지중저장 안정성 평가 파이프라인의 1단계.
지질 실현을 만들고, 그에 대한 '독립적 참값'을 유한차분으로 풀고,
강화학습 주입 제어를 위한 Gymnasium 환경을 제공한다.

원본 코드 대비 수정 내역 (감사 태그)
------------------------------------
[m1]  min-max 정규화에 의해 상쇄되던 무의미한 아핀 변환 제거
[m19] 등방 sigma=1.0 -> 층서 구조를 반영한 이방성 상관거리 + 격자 간격 명시
[m2]  반사계수를 공극률 기울기가 아닌 음향 임피던스 Z=rho*v 로부터 정식 계산(부호 포함)
[M17] 깊이축 -> 양방향 주시(TWT) 변환 후 시간영역에서 웨이브렛 컨볼루션
[m3]  리커 웨이브렛을 홀수 길이 대칭 커널로 생성(위상 이동 제거)
[m18] 전역 min-max 대신 분위수 클리핑 기반 강건 정규화
[M18] 정규화 log-투과율과 실제 투과율[mD]을 명확히 분리해 반환
[M5]  공극률과 투과율의 결정론적 종속성을 유지하되, 독립적인 KC 잔차를 별도 확률변수로 도입
[M3]  모든 물리 상수를 ReservoirConfig 에 명시하고 무차원화를 수행
[C1]  독립적 참값 생성을 위한 유한차분 기준해 솔버(FDMReference) 신규 구현
[C7]  행동이 다음 상태에 영향을 주는 진짜 MDP 로 강화학습 환경 재설계
[M16] terminated / truncated 규약 준수
[m7]  행동공간을 [-1,1] 대칭으로 정규화
[m8]  관측공간에 물리적 상·하한 부여
[m9]  reset() 마다 지질 실현을 재표집해 에피소드 다양성 확보
[M10] 두 목적 보상을 각각 [0,1] 로 정규화해 실질적 파레토 트레이드오프 확보
[m10] 환경이 계산한 수익을 info 로 노출해, 평가 코드가 별도 정의를 다시 쓰지 않게 함

범위에 대한 명시  [m20]
--------------------
이 모델은 **단상 압력확산**이다. 중력/부력항, 2상 유동, CO2 포화도, 상태방정식은
포함하지 않는다. 따라서 '부력에 의한 CO2 상승과 덮개암 하부 축적'이라는 누출의
지배 물리는 다루지 않으며, 평가 대상은 어디까지나 '주입에 의한 압력 상승'이다.
논문 서술에서 '누출 위험 평가'라는 표현을 쓰려면 이 한계를 명시해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from scipy.sparse.linalg import cg, LinearOperator

# ---------------------------------------------------------------------------
# 단위 상수
# ---------------------------------------------------------------------------
MD_TO_M2 = 9.869233e-16      # 1 milliDarcy -> m^2
SEC_PER_YEAR = 3.15576e7     # 1 year -> s
MPA = 1.0e6                  # 1 MPa -> Pa


# ---------------------------------------------------------------------------
# [M3] 물리 구성 — 모든 상수를 한 곳에 명시하고 무차원 그룹을 유도한다
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReservoirConfig:
    """저류층·유체·운영 상수. 원본 코드에 없던 차원 정보를 전부 명시한다."""

    # --- 격자 (원본에는 격자 간격 자체가 없어 라플라시안이 무차원이었다) ---
    nx: int = 64
    ny: int = 64
    nz: int = 32
    dx: float = 200.0            # m  (수평 12.8 km 광역 도메인)
    dy: float = 200.0            # m
    dz: float = 20.0             # m  (수직 640 m)

    # --- 암석/유체 물성 ---
    mu: float = 5.0e-4           # Pa*s   유체 점성도
    c_t: float = 1.0e-9          # 1/Pa   총압축률(암석+유체)
    phi_ref: float = 0.19        # -      기준 공극률
    k_ref_mD: float = 10.0       # mD     기준 투과율
    kv_kh: float = 0.10          # -      수직/수평 투과율비(층리 이방성)
    rho_matrix: float = 2650.0   # kg/m3  석영 골격
    rho_fluid: float = 1030.0    # kg/m3  염수
    v_matrix: float = 5500.0     # m/s    골격 P파 속도
    v_fluid: float = 1500.0      # m/s    유체 P파 속도

    # --- 압력 기준 ---
    p_init: float = 30.0 * MPA   # Pa  초기 저류층 압력
    p_limit: float = 45.0 * MPA  # Pa  운영 임계압력 (= 1.5 * p_init)
    p_hard: float = 50.0 * MPA   # Pa  즉시 중단 기준
    dp_ref: float = 15.0 * MPA   # Pa  특성 과압 (무차원화 기준)

    # --- 운영 ---
    # 2.5 Mt/yr 에서는 최대 주입(a=1.5)에서도 P97.5 가 43.3 MPa 로 임계 45 MPa 에
    # 도달하지 못해 안전 제약이 비활성이 된다. Gorgon 급(약 4 Mt/yr) 범위 안에서
    # 제약이 꼬리 구간에서만 구속되도록 3.5 로 설정한다.
    inject_mt_per_year: float = 3.5   # Mt-CO2/yr  공칭 주입량
    rho_co2: float = 700.0            # kg/m3      저류층 조건 CO2 밀도
    inj_years: float = 30.0           # yr         주입 기간
    horizon_years: float = 100.0      # yr         모니터링 종료
    carbon_price0: float = 50.0       # USD/tCO2
    carbon_price_growth: float = 0.05 # 1/yr

    # --- 탄성파 취득 ---
    seismic_freq_hz: float = 25.0
    seismic_dt_s: float = 0.002

    # ---------------- 유도량 ----------------
    @property
    def k_ref(self) -> float:
        return self.k_ref_mD * MD_TO_M2

    @property
    def extent(self) -> tuple[float, float, float]:
        return self.nx * self.dx, self.ny * self.dy, self.nz * self.dz

    @property
    def half_extent(self) -> tuple[float, float, float]:
        lx, ly, lz = self.extent
        return lx / 2.0, ly / 2.0, lz / 2.0

    @property
    def eta_ref(self) -> float:
        """기준 수압확산계수 eta = k / (phi mu c_t)   [m^2/s]"""
        return self.k_ref / (self.phi_ref * self.mu * self.c_t)

    def diffusion_coeffs(self, phi_mean: float) -> tuple[float, float, float]:
        """정규화 좌표 xi in [-1,1] 에서의 무차원 확산 계수 (A_x, A_y, A_z).

        무차원화:  x = H_i * xi,  t = T * t*,  p = p_init + dp_ref * p*
            phi c_t dp/dt = div( (k/mu) grad p )
        ->  dp*/dt* = sum_i A_i d/dxi_i ( k* dp*/dxi_i ),   k* = k / k_ref
        with A_i = eta_ref * T / H_i^2 * (phi_ref / phi)
        """
        hx, hy, hz = self.half_extent
        scale = self.eta_ref * SEC_PER_YEAR * (self.phi_ref / phi_mean)
        return scale / hx**2, scale / hy**2, scale / hz**2 * self.kv_kh

    def source_strength(self, phi_mean: float) -> float:
        """셀 하나에 전량 주입할 때의 무차원 소스 세기 [1/yr]."""
        q_vol = self.inject_mt_per_year * 1e9 / self.rho_co2 / SEC_PER_YEAR  # m^3/s
        cell_volume = self.dx * self.dy * self.dz
        return (q_vol * SEC_PER_YEAR) / (phi_mean * self.c_t * self.dp_ref * cell_volume)

    def carbon_price(self, year):
        return self.carbon_price0 * (1.0 + self.carbon_price_growth) ** year

    def to_mpa(self, p_star):
        """무차원 과압 p* -> 절대압력 [MPa]"""
        return (self.p_init + p_star * self.dp_ref) / MPA


# ---------------------------------------------------------------------------
# [1] 리커 웨이브렛  — [m3] 홀수 길이 대칭 커널
# ---------------------------------------------------------------------------
def generate_ricker_wavelet(freq_hz: float, dt_s: float, length_s: float):
    """영위상 리커 웨이브렛.  w(t) = (1 - 2 pi^2 f^2 t^2) exp(-pi^2 f^2 t^2)

    원본은 np.arange(-L/2, L/2, dt) 로 짝수 길이(50) 커널을 만들어
    mode='same' 컨볼루션에서 1샘플 위상 이동이 발생했다.
    여기서는 홀수 길이를 강제해 중심 샘플이 정확히 t=0 이 되도록 한다.
    """
    half = int(round(length_s / 2.0 / dt_s))
    t = np.arange(-half, half + 1) * dt_s          # 길이 2*half+1 (홀수)
    a = (np.pi * freq_hz * t) ** 2
    wavelet = (1.0 - 2.0 * a) * np.exp(-a)
    return wavelet, t


def _robust_normalize(data, lo_q=0.005, hi_q=0.995, out_range=(0.0, 1.0)):
    """[m18] 전역 min/max 대신 분위수 클리핑 후 선형 사상."""
    lo, hi = np.quantile(data, lo_q), np.quantile(data, hi_q)
    if hi - lo < 1e-30:
        return np.full_like(data, 0.5 * (out_range[0] + out_range[1]))
    scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    return scaled * (out_range[1] - out_range[0]) + out_range[0]


# ---------------------------------------------------------------------------
# [2] 3D 지질 모델
# ---------------------------------------------------------------------------
@dataclass
class GeoModel:
    porosity: np.ndarray          # [-]     공극률 (nx,ny,nz)
    permeability_mD: np.ndarray   # [mD]    실제 투과율
    k_star: np.ndarray            # [-]     k / k_ref  (PDE 계수)
    perm_norm: np.ndarray         # [-]     log10 k 의 [0,1] 정규화 (시각화/보고용)
    seismic_time: np.ndarray      # [-]     시간영역 합성 탄성파 (nx,ny,nt), [-1,1]
    twt_axis: np.ndarray          # [s]     양방향 주시 축
    well_log_z: np.ndarray        # [-]     Z-score 정규화된 1D 시추로그
    phi_mean: float
    kc_residual_log10: float

    def summary(self) -> dict:
        return {
            "phi_mean": self.phi_mean,
            "kc_residual_log10": self.kc_residual_log10,
            "k_mD_median": float(np.median(self.permeability_mD)),
            "k_mD_min": float(self.permeability_mD.min()),
            "k_mD_max": float(self.permeability_mD.max()),
        }


def create_geological_model(
    cfg: ReservoirConfig,
    phi_mean: float = 0.189,
    kc_residual_log10: float = 0.0,
    corr_len_h_m: float = 800.0,
    corr_len_v_m: float = 25.0,
    rng: np.random.Generator | None = None,
) -> GeoModel:
    """상관 구조를 갖는 공극률장과 그로부터 유도된 투과율·합성 탄성파를 생성한다.

    [M9] 주의: seismic_time 과 well_log_z 는 지질 모델의 현실성 점검·시각화를 위해
    함께 반환하지만, 이 파이프라인의 PINN 입력은 (x,y,z,t,phi,eps) 뿐이므로 이들이
    학습에 쓰이지는 않는다. 원본 논문의 '지진파와 시추로그를 결합한 데이터셋으로
    학습' 이라는 서술이 성립하려면 지진파 -> 물성 역산 경로를 추가로 구현해야 한다.

    Parameters
    ----------
    phi_mean : 목표 평균 공극률.  [M4] PCE 입력 분포와 동일한 변수로 다룬다.
    kc_residual_log10 : Kozeny-Carman 관계식의 log10 잔차.
        [M5] 공극률과 투과율은 물리적으로 종속이므로 k 를 독립 변수로 두지 않고,
        KC 관계식의 산포만을 독립 확률변수 eps 로 분리한다.
        -> (phi_mean, eps) 는 서로 독립이므로 Sobol 분산분해가 정의된다.
    corr_len_h_m, corr_len_v_m : [m19] 물리 단위 상관거리(수평/수직).
    """
    rng = np.random.default_rng() if rng is None else rng
    size = (cfg.nx, cfg.ny, cfg.nz)

    # --- 상관거리를 격자 단위 sigma 로 환산 (가우시안 커널 -> 가우시안형 공분산) ---
    sigma = (
        corr_len_h_m / cfg.dx / 2.0,
        corr_len_h_m / cfg.dy / 2.0,
        corr_len_v_m / cfg.dz / 2.0,
    )
    field = gaussian_filter(rng.random(size), sigma=sigma, mode="wrap")

    # [m1] 원본의 (x-mean)*2.5+mean 는 뒤따르는 정규화에 의해 완전히 상쇄되므로 제거.
    # 표준화 후 목표 평균/표준편차로 직접 사상한다.
    field = (field - field.mean()) / (field.std() + 1e-30)
    phi_std = 0.024                                   # 논문 표 1 수준의 산포 유지
    porosity = np.clip(phi_mean + phi_std * field, 0.02, 0.38)

    # --- Kozeny-Carman + 독립 잔차 ---
    kc_A = 1000.0  # mD; 입경/비표면적/굴곡도를 흡수한 계수
    permeability_mD = kc_A * porosity**3 / (1.0 - porosity) ** 2
    permeability_mD *= 10.0 ** kc_residual_log10
    k_star = permeability_mD * MD_TO_M2 / cfg.k_ref

    # [M18] 정규화 log-투과율은 '투과율'이 아니라 보고용 무차원량임을 이름으로 구분
    perm_norm = _robust_normalize(np.log10(permeability_mD))

    # --- [m2][M17] 임피던스 기반 반사계수 + 깊이-시간 변환 후 컨볼루션 ---
    rho = porosity * cfg.rho_fluid + (1.0 - porosity) * cfg.rho_matrix
    slowness = porosity / cfg.v_fluid + (1.0 - porosity) / cfg.v_matrix   # Wyllie 시간평균
    velocity = 1.0 / slowness
    impedance = rho * velocity

    # 반사계수 R = (Z_{i+1} - Z_i) / (Z_{i+1} + Z_i).
    # 공극률이 커지면 임피던스가 작아지므로 부호가 자동으로 맞는다.
    refl = np.zeros_like(impedance)
    refl[:, :, :-1] = np.diff(impedance, axis=2) / (
        impedance[:, :, :-1] + impedance[:, :, 1:]
    )

    # 각 트레이스의 양방향 주시(TWT)
    twt = 2.0 * np.cumsum(cfg.dz / velocity, axis=2)
    twt_max = float(twt.max())
    n_t = int(np.ceil(twt_max / cfg.seismic_dt_s)) + 1
    twt_axis = np.arange(n_t) * cfg.seismic_dt_s

    refl_time = np.zeros((cfg.nx, cfg.ny, n_t))
    for i in range(cfg.nx):
        for j in range(cfg.ny):
            refl_time[i, j] = np.interp(twt_axis, twt[i, j], refl[i, j], left=0.0, right=0.0)

    wavelet, _ = generate_ricker_wavelet(
        cfg.seismic_freq_hz, cfg.seismic_dt_s, length_s=4.0 / cfg.seismic_freq_hz
    )
    seismic = fftconvolve(refl_time, wavelet[None, None, :], mode="same", axes=2)
    seismic_time = _robust_normalize(seismic, out_range=(-1.0, 1.0))

    # --- 1D 시추로그 (Z-score) ---
    well_raw = porosity[cfg.nx // 2, cfg.ny // 2, :]
    well_log_z = (well_raw - well_raw.mean()) / (well_raw.std() + 1e-30)

    return GeoModel(
        porosity=porosity,
        permeability_mD=permeability_mD,
        k_star=k_star,
        perm_norm=perm_norm,
        seismic_time=seismic_time,
        twt_axis=twt_axis,
        well_log_z=well_log_z,
        phi_mean=float(porosity.mean()),
        kc_residual_log10=float(kc_residual_log10),
    )


# ---------------------------------------------------------------------------
# [3] 기초 통계
# ---------------------------------------------------------------------------
def basic_stats(data, ddof: int = 0) -> dict:
    data = np.asarray(data)
    return {
        "Min": float(np.min(data)),
        "Max": float(np.max(data)),
        "Mean": float(np.mean(data)),
        "Std": float(np.std(data, ddof=ddof)),
        "Median": float(np.median(data)),
        "Q05": float(np.quantile(data, 0.05)),
        "Q95": float(np.quantile(data, 0.95)),
    }


# ---------------------------------------------------------------------------
# 주입정 소스항 — FDM 기준해와 PINN 이 **동일한 함수**를 쓰도록 모듈 수준에 둔다.
# 원본은 3D 등방 가우시안 블롭이었으나 주입정은 연직 선소스이며,
# 주입 종료(30년) 이후 소스가 꺼지는 시간 의존성도 원본에는 없었다.
# ---------------------------------------------------------------------------
WELL_RADIUS_CELLS = 1.2      # 수평 방향 소스 반경(격자 셀)
PERF_SMOOTH_CELLS = 0.8      # 개공 구간 경계의 완만화 폭(격자 셀)
SHUTOFF_SMOOTH_YEARS = 0.25  # 주입 종료 완만화 폭(년)
PERF_TOP_FRAC = 0.55         # 개공 구간 상단 (깊이 비율)
PERF_BOT_FRAC = 0.90         # 개공 구간 하단


def _is_torch(v) -> bool:
    return hasattr(v, "clamp") and hasattr(v, "exp")


def _sigmoid(v):
    """numpy / torch 양쪽에서 동작하는 시그모이드."""
    if _is_torch(v):
        return 1.0 / (1.0 + (-v.clamp(-60.0, 60.0)).exp())
    return 1.0 / (1.0 + np.exp(-np.clip(v, -60.0, 60.0)))


def _exp(v):
    return v.exp() if _is_torch(v) else np.exp(v)


def source_shape_grid(cfg: ReservoirConfig, ix, iy, iz):
    """연속 격자 인덱스 좌표에서 평가한 주입정 형상함수(정규화 전).

    수평: 좁은 가우시안(연직정),  수직: 하부 개공 구간의 매끄러운 창함수.
    잔차 계산에서 미분 가능해야 하므로 부울 마스크 대신 시그모이드 곱을 쓴다.
    """
    r2 = (ix - cfg.nx / 2.0 + 0.5) ** 2 + (iy - cfg.ny / 2.0 + 0.5) ** 2
    radial = _exp(-r2 / (2.0 * WELL_RADIUS_CELLS**2))
    lo, hi = cfg.nz * PERF_TOP_FRAC, cfg.nz * PERF_BOT_FRAC
    window = _sigmoid((iz - lo) / PERF_SMOOTH_CELLS) * _sigmoid((hi - iz) / PERF_SMOOTH_CELLS)
    return radial * window


def injection_schedule(cfg: ReservoirConfig, t_years):
    """주입 종료 시점에서 매끄럽게 0 으로 떨어지는 스케줄 (미분 가능)."""
    return _sigmoid((cfg.inj_years - t_years) / SHUTOFF_SMOOTH_YEARS)


def injection_schedule_mean(cfg: ReservoirConfig, t0: float, t1: float) -> float:
    """[t0, t1] 구간에서 injection_schedule 의 **정확한** 시간 평균.

    시간 스텝이 종료 전이폭(0.25 yr)보다 크면 중점값 injection_schedule((t0+t1)/2)
    은 스케줄을 심하게 잘못 표현한다(스텝이 30년을 걸치면 '30년까지 100% 주입'으로
    읽힌다). 시그모이드의 원시함수가 softplus 이므로 해석적으로 평균을 구한다.

        d/dt [ -w * softplus((T-t)/w) ] = sigmoid((T-t)/w)
    """
    if t1 - t0 < 1e-12:
        return float(injection_schedule(cfg, t0))
    w = SHUTOFF_SMOOTH_YEARS
    sp = lambda u: np.logaddexp(0.0, u)
    u0, u1 = (cfg.inj_years - t0) / w, (cfg.inj_years - t1) / w
    return float(w * (sp(u0) - sp(u1)) / (t1 - t0))


def source_normalizer(cfg: ReservoirConfig) -> float:
    """격자 셀 합이 1 이 되도록 하는 형상함수 정규화 상수."""
    ix, iy, iz = np.meshgrid(
        np.arange(cfg.nx), np.arange(cfg.ny), np.arange(cfg.nz), indexing="ij"
    )
    return float(source_shape_grid(cfg, ix, iy, iz).sum())


# ---------------------------------------------------------------------------
# [4] [C1] 유한차분 기준해 솔버 — PINN 검증을 위한 '독립적 참값' 생성기
# ---------------------------------------------------------------------------
class FDMReference:
    """무차원 압력확산 방정식의 유한차분 기준해.

        dp*/dt* = sum_i A_i d/dxi_i ( k* dp*/dxi_i ) + s*(xi, t*)

    - 면 투과율은 조화평균(이질 매질의 보존적 이산화)
    - 측면: 원거리 정압(Dirichlet p*=0) — 저류층이 도메인 밖으로 이어짐
    - 상/하부: 무유동(Neumann) — 덮개암과 하부 차수층
    - 시간적분: 후진 오일러(무조건 안정) + 야코비 전처리 CG

    원본 코드에는 참값 생성기가 전혀 없었고, 검증 시 모델 자신의 예측에
    잡음을 더해 '관측치'로 사용했다(C1). 이 클래스가 그 순환을 끊는다.
    """

    def __init__(self, cfg: ReservoirConfig, geo: GeoModel):
        self.cfg = cfg
        self.geo = geo
        nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
        self.shape = (nx, ny, nz)
        self.n = nx * ny * nz

        ax, ay, az = cfg.diffusion_coeffs(geo.phi_mean)
        hx, hy, hz = 2.0 / nx, 2.0 / ny, 2.0 / nz
        k = geo.k_star

        def harmonic(a, axis):
            s1 = [slice(None)] * 3
            s2 = [slice(None)] * 3
            s1[axis] = slice(0, -1)
            s2[axis] = slice(1, None)
            lo, hi = a[tuple(s1)], a[tuple(s2)]
            return 2.0 * lo * hi / (lo + hi + 1e-30)

        self.tx = ax * harmonic(k, 0) / hx**2
        self.ty = ay * harmonic(k, 1) / hy**2
        self.tz = az * harmonic(k, 2) / hz**2
        # 측면 Dirichlet: 경계까지의 거리는 반쪽 셀
        self.bx0 = ax * k[0] / (hx**2 / 2.0)
        self.bx1 = ax * k[-1] / (hx**2 / 2.0)
        self.by0 = ay * k[:, 0] / (hy**2 / 2.0)
        self.by1 = ay * k[:, -1] / (hy**2 / 2.0)

        diag = np.zeros(self.shape)
        diag[:-1] += self.tx
        diag[1:] += self.tx
        diag[:, :-1] += self.ty
        diag[:, 1:] += self.ty
        diag[:, :, :-1] += self.tz
        diag[:, :, 1:] += self.tz
        diag[0] += self.bx0
        diag[-1] += self.bx1
        diag[:, 0] += self.by0
        diag[:, -1] += self.by1
        self.diag = diag.ravel()

        self.source = self._build_source()

    # ------------------------------------------------------------------
    def _build_source(self) -> np.ndarray:
        """PINN 잔차와 **동일한** 형상함수로부터 격자 소스 벡터를 만든다.

        원본은 3D 등방 가우시안 블롭이었으나 주입정은 연직 선소스이다.
        """
        cfg = self.cfg
        ix, iy, iz = np.meshgrid(
            np.arange(cfg.nx), np.arange(cfg.ny), np.arange(cfg.nz), indexing="ij"
        )
        shape = source_shape_grid(cfg, ix, iy, iz) / source_normalizer(cfg)
        return (cfg.source_strength(self.geo.phi_mean) * shape).ravel()

    def _apply_operator(self, vec: np.ndarray) -> np.ndarray:
        """div(k* grad p*) 를 적용."""
        p = vec.reshape(self.shape)
        out = np.zeros_like(p)
        f = self.tx * (p[1:] - p[:-1]); out[:-1] += f; out[1:] -= f
        f = self.ty * (p[:, 1:] - p[:, :-1]); out[:, :-1] += f; out[:, 1:] -= f
        f = self.tz * (p[:, :, 1:] - p[:, :, :-1]); out[:, :, :-1] += f; out[:, :, 1:] -= f
        out[0] -= self.bx0 * p[0]
        out[-1] -= self.bx1 * p[-1]
        out[:, 0] -= self.by0 * p[:, 0]
        out[:, -1] -= self.by1 * p[:, -1]
        return out.ravel()

    # ------------------------------------------------------------------
    def build_time_grid(self, tmax: float, dt_scale: float = 1.0) -> np.ndarray:
        """시간 적분 격자.

        후진 오일러는 시간에 대해 1차 정확도뿐이므로 스텝 크기가 곧 오차다.
        특히 (a) 초기 급상승, (b) 주입 종료 전이(폭 0.25 yr), (c) 종료 직후의
        급감압 세 구간을 해상하지 못하면 30년 압력이 수 MPa 단위로 틀어진다.
        dt_scale 을 0.5 로 주면 모든 스텝이 절반이 되어 수렴성 확인에 쓸 수 있다.
        """
        cfg = self.cfg
        s = float(dt_scale)
        pieces = [np.array([0.0]), np.geomspace(0.02, min(1.0, tmax), max(int(10 / s), 2))]
        if tmax > 1.0:
            pieces.append(np.arange(1.0, min(cfg.inj_years, tmax) + 1e-9, 1.0 * s))
        lo, hi = cfg.inj_years - 1.0, min(cfg.inj_years + 1.0, tmax)
        if hi > lo:
            pieces.append(np.arange(lo, hi + 1e-9, 0.1 * s))
        if tmax > cfg.inj_years:
            # 주입 종료 직후 감압이 가장 빠르므로 등급화한 격자를 쓴다.
            # 균일 2년 스텝을 쓰면 후진 오일러가 첫 스텝에서 압력을 통째로 무너뜨린다.
            span = tmax - cfg.inj_years
            pieces.append(cfg.inj_years
                          + np.geomspace(min(0.05, span), span, max(int(30 / s), 4)))
        grid = np.unique(np.concatenate(pieces))
        return grid[(grid >= 0.0) & (grid <= tmax + 1e-9)]

    def solve(
        self,
        output_years: np.ndarray,
        dt_scale: float = 1.0,
        rtol: float = 1e-9,
    ) -> np.ndarray:
        """지정한 시점들에서의 무차원 압력장 스냅샷을 반환한다. shape (n_out, nx, ny, nz).

        주입률은 각 스텝 구간에서 **해석적 시간 평균**을 쓴다. 중점 표본을 쓰면
        스텝이 종료 전이폭보다 클 때 주입량이 계통적으로 과대 계상된다.
        """
        output_years = np.asarray(output_years, dtype=float)
        tmax = float(output_years.max())
        grid = np.unique(np.concatenate([self.build_time_grid(tmax, dt_scale), output_years]))
        grid = grid[(grid >= 0.0) & (grid <= tmax + 1e-9)]

        p = np.zeros(self.n)
        snapshots, out_idx = [], 0
        want = np.sort(output_years)
        if want[0] <= 1e-12:
            snapshots.append(p.reshape(self.shape).copy())
            out_idx = 1

        for t0, t1 in zip(grid[:-1], grid[1:]):
            dt = t1 - t0
            if dt <= 0:
                continue
            rate = injection_schedule_mean(self.cfg, t0, t1)
            rhs = p + dt * rate * self.source
            operator = LinearOperator(
                (self.n, self.n), matvec=lambda v, dt=dt: v - dt * self._apply_operator(v),
                dtype=np.float64,
            )
            precond = LinearOperator(
                (self.n, self.n), matvec=lambda v, dt=dt: v / (1.0 + dt * self.diag),
                dtype=np.float64,
            )
            p, info = cg(operator, rhs, x0=p, rtol=rtol, maxiter=1500, M=precond)
            if info != 0:
                raise RuntimeError(f"CG가 수렴하지 않았습니다 (t={t1:.3f} yr, info={info})")
            while out_idx < want.size and want[out_idx] <= t1 + 1e-9:
                snapshots.append(p.reshape(self.shape).copy())
                out_idx += 1

        while len(snapshots) < want.size:
            snapshots.append(p.reshape(self.shape).copy())
        stacked = np.stack(snapshots)
        # want 은 정렬된 순서이므로 호출자가 준 output_years 순서로 되돌린다
        order = np.argsort(np.argsort(output_years))
        return stacked[order]

    def convergence_check(self, probe_years: np.ndarray, dt_scales=(1.0, 0.5)) -> dict:
        """시간 스텝을 절반으로 줄여도 해가 바뀌지 않는지 확인한다.

        원본에는 이런 검증이 전혀 없었고, 여기서도 이 점검을 넣기 전에는
        30년 압력이 스텝 세분에 따라 수 MPa 씩 움직였다.
        """
        sols = [self.solve(probe_years, dt_scale=s) for s in dt_scales]
        ref = sols[-1]
        return {
            "dt_scales": list(dt_scales),
            "peak_mpa": [float(self.cfg.to_mpa(s.max())) for s in sols],
            "rel_change": float(np.abs(sols[0] - ref).max() / (np.abs(ref).max() + 1e-30)),
        }


# ---------------------------------------------------------------------------
# [5] [C7] 강화학습 주입 제어 환경 — 진짜 MDP
# ---------------------------------------------------------------------------
@dataclass
class Realization:
    """한 에피소드가 마주하는 저류층 실현."""
    p_base: np.ndarray   # 공칭 주입(a=1) 하의 무차원 과압 궤적, 길이 n_years+1
    sigma: np.ndarray    # 같은 시점의 무차원 불확실성 표준편차
    phi_mean: float
    kc_residual: float


class CCSInjectionEnv(gym.Env):
    """CO2 주입률 제어 환경.

    원본 대비 결정적 차이
    ---------------------
    원본에서는 step() 이 self.current_year 만 증가시켰고 관측값은 PCE 배열을
    그대로 읽었으므로, 행동이 다음 상태에 미치는 영향이 정확히 0이었다(C7).
    즉 MDP 가 아니라 문맥적 밴딧이었고 최적해가 닫힌 형태로 존재했다.

    여기서는 압력을 내부 상태로 승격하고 다음 축약 동역학을 적분한다.

        dp/dt = a(t) * b(t) - lambda * (p - p0)

    b(t) 는 '공칭 주입에서 기준 궤적 p_base 를 정확히 재현'하도록 역산되므로
    (a == 1 이면 p == p_base), 물리 대리모델과의 정합성이 유지되면서도
    행동이 상태에 누적적으로 반영된다.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: ReservoirConfig,
        realization_sampler: Callable[[np.random.Generator], Realization],
        weight_econ: float = 0.5,
        dissipation_per_year: float = 0.04,
    ):
        super().__init__()
        self.cfg = cfg
        self.sample_realization = realization_sampler
        self.w_econ = float(weight_econ)
        self.w_safe = 1.0 - self.w_econ
        self.lam = float(dissipation_per_year)
        self.n_years = int(cfg.inj_years)

        # [m7] 대칭 정규화 행동공간. 물리 조절율은 내부에서 1.0 + 0.5*a 로 환산.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # [m8] 각 성분에 물리적 상·하한을 부여
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, -5.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 6.0, 4.0, 3.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )

        # 보상 정규화 상수 [M10]
        self._price_max = cfg.carbon_price(self.n_years - 1)
        self._econ_max = self._price_max * 1.5
        self._safe_scale = ((cfg.p_hard - cfg.p_limit) / cfg.dp_ref) ** 2

        self.realization: Realization | None = None
        self.year = 0
        self.p_star = 0.0
        self.cum_injected = 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def action_to_rate(action) -> float:
        """[-1,1] -> [0.5,1.5] 주입 조절율."""
        return 1.0 + 0.5 * float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))

    def _drive(self, year: int) -> float:
        """a == 1 일 때 기준 궤적을 재현하도록 역산된 구동항 b(t)."""
        r = self.realization
        return (r.p_base[year + 1] - r.p_base[year]) + self.lam * r.p_base[year]

    def _obs(self) -> np.ndarray:
        cfg = self.cfg
        r = self.realization
        p975 = self.p_star + 1.96 * r.sigma[self.year]
        margin = ((cfg.p_limit - cfg.p_init) / cfg.dp_ref) - p975
        obs = np.array(
            [
                self.year / self.n_years,
                self.p_star,
                cfg.carbon_price(self.year) / self._price_max,
                margin,
                self.cum_injected / self.n_years,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # [m9] 에피소드마다 새로운 지질 실현 -> 정책이 일반화할 대상이 생긴다
        self.realization = self.sample_realization(self.np_random)
        self.year = 0
        self.p_star = 0.0          # 무차원 과압 (절대압 = p_init + p_star*dp_ref)
        self.cum_injected = 0.0
        return self._obs(), {}

    def step(self, action):
        cfg = self.cfg
        rate = self.action_to_rate(action)

        # --- 상태 전이: 행동이 압력에 누적적으로 반영된다 ---
        self.p_star = self.p_star + rate * self._drive(self.year) - self.lam * self.p_star
        self.cum_injected += rate
        self.year += 1

        p975 = self.p_star + 1.96 * self.realization.sigma[self.year]
        p_limit_star = (cfg.p_limit - cfg.p_init) / cfg.dp_ref
        p_hard_star = (cfg.p_hard - cfg.p_init) / cfg.dp_ref

        # --- [M10] 두 목적을 각각 [0,1] 로 정규화 ---
        reward_econ = (cfg.carbon_price(self.year - 1) * rate) / self._econ_max
        excess = max(0.0, p975 - p_limit_star)
        reward_safe = min(excess**2 / self._safe_scale, 1.0)
        reward = self.w_econ * reward_econ - self.w_safe * reward_safe

        # --- [M16] terminated(물리적 흡수 상태) vs truncated(시간 제한) 구분 ---
        terminated = bool(p975 > p_hard_star)
        # 관례상 두 신호는 배타적으로 둔다: 물리적 흡수 상태가 시간 제한보다 우선한다.
        truncated = bool(self.year >= self.n_years) and not terminated
        if terminated:
            reward -= 1.0

        # [m10] 평가 코드가 보상 정의를 따로 다시 구현하지 않도록 환경이 직접 노출한다.
        #       원본은 env 의 reward_econ 과 평가 루프의 profit 이 서로 다른 식이었다.
        info = {
            "p_mpa": cfg.to_mpa(self.p_star),
            "p975_mpa": cfg.to_mpa(p975),
            "rate": rate,
            "revenue_usd_per_t": cfg.carbon_price(self.year - 1) * rate,
        }
        return self._obs(), float(reward), terminated, truncated, info
