"""
Adaptive PSF Identification and Richardson-Lucy Image Restoration
for Scientific CMOS Sensors
=================================================================
References
----------
[1] Moffat A.F.J. (1969) A&A 3:455
[2] Trujillo I. et al. (2001) MNRAS 328:977
[3] Richardson W.H. (1972) JOSA 62:55
[4] Lucy L.B. (1974) AJ 79:745
[5] Santos A. et al. (1997) J. Microscopy 188:264
[6] Starck J.-L., Pantin E., Murtagh F. (2002) PASP 114:1051
[7] Born M., Wolf E. (1999) Principles of Optics, 7th ed.
"""

from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Literal, Optional

from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve
from scipy.stats import kurtosis
from scipy.special import j1
from skimage import filters

try:
    from astropy.io import fits as _fits
    _ASTROPY = True
except ImportError:
    _ASTROPY = False


# ================================================================
# Domain identifiers
# ================================================================

DOMAIN_DEEP_SPACE = 'deep_space'
DOMAIN_PLANETARY  = 'planetary'
DOMAIN_MICRO      = 'microscopy'
DOMAIN_REMOTE     = 'remote'

DOMAIN_LABELS = {
    DOMAIN_DEEP_SPACE: 'Знімки глибокого космосу',
    DOMAIN_PLANETARY:  'Місяць та планетарна зйомка',
    DOMAIN_MICRO:      'Біомедична мікроскопія',
    DOMAIN_REMOTE:     'Дистанційне зондування Землі',
}

DOMAIN_KEYS    = list(DOMAIN_LABELS.keys())
DOMAIN_DISPLAY = list(DOMAIN_LABELS.values())


# ================================================================
# Configuration
# ================================================================

@dataclass
class EngineConfig:
    """
    Pipeline parameters.

    domain               Scientific domain key (DOMAIN_* constant).
    psf_model            'auto' or explicit model name.
    psf_size             Kernel side (odd). None -> auto.
    moffat_beta          Moffat beta exponent. None -> derived from kurtosis.
    gamma_bounds         (lo, hi) for scale search. None -> data-driven.
    auto_iterations      True -> adaptive RL stopping via convergence test.
    rl_iterations        Max (auto=True) or exact (auto=False) RL count.
    rl_filter_eps        RL division floor.
    rl_min_gain_frac     Convergence threshold per iteration. None -> 0.003.
                         0.001 for planetary: more iters for crater detail.
    rl_ring_max          Ringing Score ceiling that stops RL. None -> 7.5.
                         6.0 for deep_space: stops before visible star rings.
    motion_angle         Explicit motion angle (deg). None -> Fourier detect.
    sharpen_alpha        Post-deconvolution sharpening. None=adaptive, 0=off.
    independent_channels True -> per-band RL. False -> CIE-Lab luminance only.
    verbose              Print diagnostics.
    """
    domain:               str     = 'auto'
    psf_model:            Literal['auto','gaussian','cauchy','airy',
                                   'moffat','motion','scattering'] = 'auto'
    psf_size:             Optional[int]   = None
    moffat_beta:          Optional[float] = None
    gamma_bounds:         Optional[tuple] = None
    auto_iterations:      bool            = True
    rl_iterations:        int             = 30
    rl_filter_eps:        float           = 1e-6
    rl_min_gain_frac:     Optional[float] = None
    rl_ring_max:          Optional[float] = None   # ЗМІНА 4
    motion_angle:         Optional[float] = None
    sharpen_alpha:        Optional[float] = None
    independent_channels: bool            = False
    verbose:              bool            = True


def domain_config(domain: str, **overrides) -> EngineConfig:
    """
    Return EngineConfig with scientifically motivated defaults.

    deep_space : Moffat PSF (auto -> _beta_from_kurtosis_ds); no sharpening;
                 30 iters; per-channel; lambda_ring=0.40; rl_ring_max=6.0
                 (зупинка RL до появи видимих кілець навколо зірок).

    planetary  : auto PSF (κ-tree + guards: Cauchy->Moffat, beta>=3.0);
                 gamma ceil 4.5 px; lambda_detail=0.50; min_gain_frac=0.001.

    microscopy : Airy disk; tight gamma (<4 px); 50 iters; per-channel.

    remote     : auto PSF (κ-tree + guards: Cauchy->Moffat, beta>=3.5);
                 gamma ceil 6 px; muted sharpening (x2.0 slope, clip 1.2).
    """
    bases = {
        DOMAIN_DEEP_SPACE: dict(psf_model='auto',  rl_iterations=30,
                                sharpen_alpha=0.0, independent_channels=True,
                                rl_ring_max=6.0),          # ЗМІНА 4
        DOMAIN_PLANETARY:  dict(psf_model='auto',  rl_iterations=30,
                                sharpen_alpha=None, independent_channels=False,
                                rl_min_gain_frac=0.001),
        DOMAIN_MICRO:      dict(psf_model='airy',  rl_iterations=50,
                                sharpen_alpha=None, independent_channels=True),
        DOMAIN_REMOTE:     dict(psf_model='auto',  rl_iterations=25,
                                sharpen_alpha=None, independent_channels=True),
    }
    base = dict(domain=domain, auto_iterations=True)
    base.update(bases.get(domain, {}))
    base.update(overrides)
    valid = EngineConfig.__dataclass_fields__
    return EngineConfig(**{k: v for k, v in base.items() if k in valid})


# ================================================================
# PSF Kernels  (normalised: sum = 1, flux-conserving)
# ================================================================

class PSFKernels:

    @staticmethod
    def _grid(size: int):
        ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float64)
        return np.meshgrid(ax, ax)

    @classmethod
    def gaussian(cls, size: int, sigma: float) -> np.ndarray:
        """Isotropic Gaussian - atmospheric/optical blur. Eq.(2.3)."""
        xx, yy = cls._grid(size)
        k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        return (k / k.sum()).astype(np.float32)

    @classmethod
    def cauchy(cls, size: int, gamma: float) -> np.ndarray:
        """Cauchy-Lorentz - heavy-tailed turbulence. Eq.(2.4)."""
        xx, yy = cls._grid(size)
        k = (1.0 + (xx**2 + yy**2) / gamma**2) ** -1.5
        return (k / k.sum()).astype(np.float32)

    @classmethod
    def airy(cls, size: int, radius: float) -> np.ndarray:
        """Airy disk - diffraction-limited [7]. Eq.(2.5)."""
        xx, yy = cls._grid(size)
        r = np.sqrt(xx**2 + yy**2)
        with np.errstate(invalid='ignore', divide='ignore'):
            arg = np.pi * r / radius
            k = np.where(r == 0, 1.0, (2.0 * j1(arg) / arg) ** 2)
        return (k / k.sum()).astype(np.float32)

    @classmethod
    def moffat(cls, size: int, alpha: float, beta: float = 3.5) -> np.ndarray:
        """Moffat profile for atmospheric seeing [1,2]. Eq.(2.6)."""
        xx, yy = cls._grid(size)
        k = (1.0 + (xx**2 + yy**2) / alpha**2) ** (-beta)
        return (k / k.sum()).astype(np.float32)

    @staticmethod
    def motion(size: int, length: float, angle: float = 0.0) -> np.ndarray:
        """1-D motion-blur kernel. Eq.(2.7)."""
        kernel = np.zeros((size, size), dtype=np.float64)
        c = size // 2
        ca = np.cos(np.deg2rad(angle))
        sa = np.sin(np.deg2rad(angle))
        for t in np.linspace(-length / 2, length / 2, max(int(4 * length), 8)):
            xi = int(round(c + t * ca))
            yi = int(round(c + t * sa))
            if 0 <= xi < size and 0 <= yi < size:
                kernel[yi, xi] = 1.0
        if kernel.sum() == 0:
            kernel[c, c] = 1.0
        return (kernel / kernel.sum()).astype(np.float32)

    @classmethod
    def scattering(cls, size: int, k: float) -> np.ndarray:
        """Exponential scattering - BSI sensors, near-IR. Eq.(2.8)."""
        xx, yy = cls._grid(size)
        kv = np.exp(-k * np.sqrt(xx**2 + yy**2))
        return (kv / kv.sum()).astype(np.float32)

    @classmethod
    def build(cls, model: str, size: int, gamma: float,
              beta: float = 3.5, angle: float = 0.0) -> np.ndarray:
        if model == 'moffat':
            return cls.moffat(size, gamma, beta)
        if model == 'motion':
            return cls.motion(size, gamma, angle)
        return getattr(cls, model)(size, gamma)


# ================================================================
# Sharpness Metrics
# ================================================================

class SharpnessMetrics:

    @staticmethod
    def tenengrad(img: np.ndarray) -> float:
        """Mean squared Sobel gradient - primary target [5]. Eq.(2.9)."""
        f  = img.astype(np.float64)
        gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(gx**2 + gy**2))

    @staticmethod
    def laplacian_variance(img: np.ndarray) -> float:
        """Variance of Laplacian - fine-texture metric. Eq.(2.10)."""
        return float(cv2.Laplacian(img.astype(np.float64), cv2.CV_64F).var())

    @staticmethod
    def ringing_score(img: np.ndarray) -> float:
        """
        Pearson kurtosis of Laplacian - ringing detector [6]. Eq.(2.11).
        Gaussian baseline ~3.0; ringing onset > 8.0.
        """
        lap = cv2.Laplacian(img.astype(np.float64), cv2.CV_64F).ravel()
        return float(kurtosis(lap, fisher=False))

    @staticmethod
    def directional_ratio(img: np.ndarray) -> float:
        """Gradient asymmetry - motion-blur indicator. Eq.(2.12)."""
        f  = img.astype(np.float64)
        mx = float(np.mean(np.abs(cv2.Sobel(f, cv2.CV_64F, 1, 0))))
        my = float(np.mean(np.abs(cv2.Sobel(f, cv2.CV_64F, 0, 1))))
        return max(mx, my) / (min(mx, my) + 1e-9)

    @staticmethod
    def dominant_motion_angle(img: np.ndarray) -> float:
        """Fourier inertia-tensor orientation estimate (degrees)."""
        F   = np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))
        mag = np.log1p(np.abs(F))
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        r_dc = max(3, min(h, w) // 16)
        yy, xx = np.ogrid[:h, :w]
        mag[(yy - cy)**2 + (xx - cx)**2 < r_dc**2] = 0.0
        thr = np.percentile(mag[mag > 0], 95)
        ys, xs = np.where(mag >= thr)
        if len(xs) < 10:
            return 0.0
        w_ = mag[ys, xs]
        ws = w_.sum() + 1e-12
        xc  = (w_ * (xs - cx)).sum() / ws
        yc  = (w_ * (ys - cy)).sum() / ws
        cxx = (w_ * (xs - cx - xc)**2).sum() / ws
        cyy = (w_ * (ys - cy - yc)**2).sum() / ws
        cxy = (w_ * (xs - cx - xc) * (ys - cy - yc)).sum() / ws
        return float((0.5 * np.degrees(np.arctan2(2 * cxy, cxx - cyy)) + 90) % 180)

    @staticmethod
    def gradient_kurtosis(img: np.ndarray) -> float:
        """Fisher excess kurtosis of Sobel gradients - PSF tail proxy. Eq.(2.13)."""
        f  = img.astype(np.float64)
        gx = cv2.Sobel(f, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(f, cv2.CV_64F, 0, 1)
        return float(kurtosis(np.concatenate([gx.ravel(), gy.ravel()]),
                               fisher=True))


# ================================================================
# Blur Scale Estimator
# ================================================================

def estimate_blur_scale(img: np.ndarray) -> float:
    """
    Estimate PSF spatial scale (sigma-equivalent, pixels) from the
    high-frequency energy fraction of the power spectrum.

    Returns sigma_est in pixels, clipped to [0.3, 10.0].

    Note: for high-SNR planetary images with sharp limbs and craters the
    estimator may over-estimate sigma because HF energy from the object's
    own edges inflates the numerator.  Domain-specific gamma ceilings in
    _gamma_bounds compensate for this systematic bias.
    """
    F     = np.fft.fft2(img.astype(np.float64))
    power = np.abs(F) ** 2
    h, w  = power.shape
    fy    = np.fft.fftfreq(h)
    fx    = np.fft.fftfreq(w)
    fx2d, fy2d = np.meshgrid(fx, fy)
    fr = np.sqrt(fx2d**2 + fy2d**2)

    lo_mask  = fr < 0.02
    power_nd = power.copy()
    power_nd[lo_mask] = 0.0
    total = power_nd.sum() + 1e-12

    hi_frac   = float(power_nd[(fr >= 0.20) & (fr <= 0.50)].sum() / total)
    sigma_est = float(np.clip(0.20 / (hi_frac + 0.015), 0.3, 10.0))
    return sigma_est


# ================================================================
# PSF Selector
# ================================================================

class PSFSelector:
    """
    Automatic PSF model identification (Section 2.3 of thesis).

    Decision tree (domain-aware)
    ----------------------------
    1.  D > threshold               -> Motion blur (Fourier angle detection)
    2.  domain == deep_space        -> Moffat + _beta_from_kurtosis_ds  [ЗМІНА 3]
    3.  domain == microscopy        -> Airy disk [7]
    4.  domain in {planetary,remote}-> κ-tree + physical guards         [ЗМІНА 5]
    5.  stellar heuristic           -> Moffat (generic path)
    6.  κ > 10                      -> Cauchy
    7.  κ > 5                       -> Moffat
    8.  κ > 1.5                     -> Airy
    9.  else                        -> Gaussian

    ЗМІНА 3 — _beta_from_kurtosis_ds (deep_space only):
      κ > 12 → β = 7.0  Moffat ≈ Gaussian; models brighter-fatter PSF (§1.3).
               At high fill-factor the PSF widens toward a Gaussian profile,
               and β = 7.0 suppresses heavy-tail ringing without losing
               the Moffat flux-conservation guarantee.
      κ > 10 → β = 2.5  Heavy tails (strong turbulence / crosstalk)
      κ >  7 → β = 3.0
      κ >  4 → β = 3.5
       else  → β = 4.0  Mild tails (stable seeing)

    ЗМІНА 5 — physical guards for planetary / remote:
      Planetary: Cauchy (κ>10) is redirected to Moffat; beta = max(beta, 3.0).
        Motivation: lunar/planetary PSF ≈ Moffat β 3–4 [2]; Cauchy (β≈1)
        over-deconvolves crater edges and amplifies ringing.
      Remote:    Cauchy redirected to Moffat; beta = max(beta, 3.5).
        Motivation: satellite PSF = atmosphere + platform vibration ≈ Moffat
        β 3.5–5.0; Gaussian is retained as the κ≤1.5 branch (now metric-driven
        instead of hard-coded).
    """

    @staticmethod
    def _is_stellar(img: np.ndarray) -> bool:
        thr     = np.percentile(img, 15)
        bg_frac = float(np.mean(img <= thr))
        bg_med  = float(np.median(img[img <= thr]) + 1e-9)
        return bg_frac >= 0.70 and float(img.max()) / bg_med >= 50.0

    @staticmethod
    def _beta_from_kurtosis(k: float) -> float:
        """General adaptive Moffat beta (non-deep_space paths) [2]."""
        if k > 10: return 2.5
        if k > 7:  return 3.0
        if k > 4:  return 3.5
        return 4.0

    @staticmethod
    def _beta_from_kurtosis_ds(k: float) -> float:
        """
        ЗМІНА 3: Deep-space beta lookup with brighter-fatter extension.

        κ > 12 → β = 7.0  (Moffat near-Gaussian limit, brighter-fatter §1.3)
        κ > 10 → β = 2.5  (heavy tails)
        κ >  7 → β = 3.0
        κ >  4 → β = 3.5
         else  → β = 4.0
        """
        if k > 12: return 7.0
        if k > 10: return 2.5
        if k > 7:  return 3.0
        if k > 4:  return 3.5
        return 4.0

    @classmethod
    def select(cls, img: np.ndarray, domain: str = 'auto',
               verbose: bool = True):
        """Returns (model_str, motion_angle_or_None, moffat_beta)."""
        _v = lambda s: print(f"  [PSFSelector] {s}") if verbose else None

        dr = SharpnessMetrics.directional_ratio(img)
        _v(f"D = {dr:.3f}")
        motion_thr = 2.0 if domain == DOMAIN_REMOTE else 2.5
        if dr > motion_thr:
            angle = SharpnessMetrics.dominant_motion_angle(img)
            _v(f"-> Motion  theta={angle:.1f} deg")
            return 'motion', angle, 3.5

        k_val   = SharpnessMetrics.gradient_kurtosis(img)
        stellar = cls._is_stellar(img)
        _v(f"kappa={k_val:.2f}  stellar={stellar}  domain={domain}")

        # ── ЗМІНА 3: deep_space — always Moffat, domain-specific beta ──
        if domain == DOMAIN_DEEP_SPACE:
            beta = cls._beta_from_kurtosis_ds(k_val)
            regime = ('near-Gaussian [brighter-fatter]'
                      if beta >= 6.0 else 'heavy-tail seeing')
            _v(f"-> Moffat (deep_space beta={beta:.1f}  {regime})")
            return 'moffat', None, beta

        # ── microscopy: diffraction-limited ────────────────────────────
        if domain == DOMAIN_MICRO:
            _v("-> Airy (microscopy)")
            return 'airy', None, 3.5

        # ── ЗМІНА 5: planetary / remote — κ-tree + physical guards ─────
        if domain in (DOMAIN_PLANETARY, DOMAIN_REMOTE):
            beta_raw = cls._beta_from_kurtosis(k_val)
            if k_val > 10.0:
                # Cauchy suppressed: too heavy-tailed for these domains
                model_raw = 'moffat'
                _v(f"kappa>{10:.0f}: Cauchy -> Moffat (guard, domain={domain})")
            elif k_val > 5.0:
                model_raw = 'moffat'
            elif k_val > 1.5:
                model_raw, beta_raw = 'airy', 3.5
            else:
                model_raw, beta_raw = 'gaussian', 3.5

            beta_floor = 3.0 if domain == DOMAIN_PLANETARY else 3.5
            beta_final = max(beta_raw, beta_floor)
            if beta_final != beta_raw:
                _v(f"beta floor: {beta_raw:.1f} -> {beta_final:.1f} "
                   f"(floor={beta_floor}, domain={domain})")
            _v(f"-> {model_raw}  beta={beta_final:.1f}  ({domain} guarded)")
            return model_raw, None, beta_final

        # ── generic κ-tree (fallback for unrecognised domain) ──────────
        beta = cls._beta_from_kurtosis(k_val)
        if stellar:
            _v(f"-> Moffat (stellar heuristic, beta={beta:.1f})")
            return 'moffat', None, beta
        if k_val > 10.0:
            _v("-> Cauchy")
            return 'cauchy', None, 3.5
        if k_val > 5.0:
            _v(f"-> Moffat (beta={beta:.1f})")
            return 'moffat', None, beta
        if k_val > 1.5:
            _v("-> Airy")
            return 'airy', None, 3.5
        _v("-> Gaussian")
        return 'gaussian', None, 3.5


# ================================================================
# PSF Parameter Optimiser  (ringing- and detail-penalised)
# ================================================================

class GammaOptimizer:
    """
    Bounded scalar optimisation of PSF scale parameter γ*. Extended Eq.(2.14):

        J(γ) = −T(f̂_γ)
               + λ_ring   · max(0, R − R_ref) · T(f̂_γ)     [ringing penalty]
               + λ_detail · max(0, T(g) − T(f̂_γ))           [detail penalty]

    T = Tenengrad [5], R = Ringing Score [6].
    T(g) = Tenengrad of the input image, pre-computed in __init__.

    Domain constants
    ────────────────
    Domain        λ_ring   λ_detail   Notes
    deep_space     0.40      0.00     Strong anti-ring; stars are point sources
    planetary      0.15      0.50     ЗМІНА 2: λ_detail forces smaller γ.
                                       λ=0.50 dominates when T(f̂) < 0.67·T(g)
    others         0.15      0.00     Original behaviour

    R_ref = 5.0 (natural leptokurtosis above Gaussian baseline 3.0 [6]).

    ЗМІНА 2 rationale:
      estimate_blur_scale over-estimates σ for high-SNR planetary images
      because sharp limbs/craters contribute HF power even before deconv.
      A γ that results in T(f̂) < T(g) means the PSF was over-broad and
      blurred the image further — exactly the «soap» symptom.  The detail
      penalty λ_detail·max(0, T(g)−T(f̂)) eliminates such candidates.
    """

    _RING_REF               = 5.0
    _RING_LAMBDA_DEFAULT    = 0.15
    _RING_LAMBDA_DEEP_SPACE = 0.40
    _DETAIL_LAMBDA_PLANETARY = 0.50   # ЗМІНА 2

    def __init__(self, img: np.ndarray, model: str, size: int,
                 angle: float = 0.0, beta: float = 3.5, eps: float = 1e-6,
                 domain: str = ''):
        self.img   = img.astype(np.float64)
        self.model = model
        self.size  = size
        self.angle = angle
        self.beta  = beta
        self.eps   = eps

        self._ring_lambda = (
            self._RING_LAMBDA_DEEP_SPACE
            if domain == DOMAIN_DEEP_SPACE
            else self._RING_LAMBDA_DEFAULT
        )

        # ЗМІНА 2: pre-compute T(g) for planetary detail penalty
        if domain == DOMAIN_PLANETARY:
            self._detail_lambda = self._DETAIL_LAMBDA_PLANETARY
            self._t_input = SharpnessMetrics.tenengrad(
                np.clip(img, 0.0, 1.0).astype(np.float32))
        else:
            self._detail_lambda = 0.0
            self._t_input       = 0.0

    def _objective(self, gamma: float) -> float:
        psf  = PSFKernels.build(self.model, self.size, gamma,
                                 self.beta, self.angle).astype(np.float64)
        f32  = _rl_steps(self.img, psf, n=5, eps=self.eps)
        t    = SharpnessMetrics.tenengrad(f32)
        ring = SharpnessMetrics.ringing_score(f32)

        ring_penalty   = (max(0.0, ring - self._RING_REF)
                          * self._ring_lambda * t)
        # ЗМІНА 2: penalise gamma where deconv reduces sharpness vs input
        detail_penalty = (max(0.0, self._t_input - t)
                          * self._detail_lambda)

        return -(t - ring_penalty - detail_penalty)

    def run(self, bounds: tuple, verbose: bool = True) -> float:
        if verbose:
            print(f"  [Optimizer] model={self.model}  "
                  f"λ_ring={self._ring_lambda:.2f}  "
                  f"λ_detail={self._detail_lambda:.2f}  "
                  f"gamma in [{bounds[0]:.3f}, {bounds[1]:.3f}]")
        res = minimize_scalar(self._objective, bounds=bounds,
                              method='bounded',
                              options={'xatol': 1e-3, 'maxiter': 60})
        if verbose:
            print(f"  [Optimizer] gamma* = {res.x:.4f}   "
                  f"score = {-res.fun:.4f}")
        return float(res.x)


# ================================================================
# Richardson-Lucy core  (FFT warm-start loop)
# ================================================================

def _rl_steps(img: np.ndarray, psf: np.ndarray,
              n: int, eps: float) -> np.ndarray:
    """Run n RL iterations starting from img. Returns float32 in [0,1]."""
    psf_flip = psf[::-1, ::-1]
    f = img.copy()
    for _ in range(n):
        denom = fftconvolve(f, psf, mode='same')
        ratio = img / np.maximum(denom, eps)
        f    *= fftconvolve(ratio, psf_flip, mode='same')
        np.clip(f, 0.0, None, out=f)
    return np.clip(f, 0.0, 1.0).astype(np.float32)


def rl_adaptive(img: np.ndarray, psf: np.ndarray,
                max_iter: int, eps: float,
                min_gain_frac: float = 0.003,
                ring_max: float = 7.5,
                chunk: int = 5,
                verbose: bool = True):
    """
    Richardson-Lucy with adaptive stopping [3,4,6].

    Stopping criteria:
      1. Ringing Score > ring_max  -> artefact onset, stop immediately.
         ЗМІНА 4: ring_max is now passed from EngineConfig.rl_ring_max.
           deep_space: ring_max = 6.0  (early stop before visible star rings)
           default   : ring_max = 7.5  (original behaviour)
      2. Marginal Tenengrad gain per iteration < min_gain_frac -> convergence.
           planetary: min_gain_frac = 0.001 (more iters for crater detail)

    Returns (restored_float32, iters_done).
    """
    psf_f64  = psf.astype(np.float64)
    psf_flip = psf_f64[::-1, ::-1]
    img_f64  = img.astype(np.float64)
    f        = img_f64.copy()
    t_prev   = SharpnessMetrics.tenengrad(
                   np.clip(f, 0, 1).astype(np.float32))
    iters_done = 0

    for start in range(0, max_iter, chunk):
        n = min(chunk, max_iter - start)
        for _ in range(n):
            denom = fftconvolve(f, psf_f64, mode='same')
            ratio = img_f64 / np.maximum(denom, eps)
            f    *= fftconvolve(ratio, psf_flip, mode='same')
            np.clip(f, 0.0, None, out=f)
        iters_done += n

        f32  = np.clip(f, 0.0, 1.0).astype(np.float32)
        t    = SharpnessMetrics.tenengrad(f32)
        ring = SharpnessMetrics.ringing_score(f32)
        gain = (t - t_prev) / (abs(t_prev) + 1e-9) / chunk

        if verbose:
            print(f"  [RL] k={iters_done:3d}  T={t:.4f}  "
                  f"dT/k={gain:+.4f}  R={ring:.2f}")

        if ring > ring_max:
            if verbose:
                print(f"  [RL] Stop: ringing {ring:.2f} > {ring_max:.1f}")
            break

        if iters_done > chunk and gain < min_gain_frac:
            if verbose:
                print(f"  [RL] Converged at k={iters_done}")
            break

        t_prev = t

    return np.clip(f, 0.0, 1.0).astype(np.float32), iters_done


# ================================================================
# Adaptive Multiscale Sharpening
# ================================================================

class AdaptiveSharpen:
    """
    Post-deconvolution multiscale unsharp masking [Eq. 2.16-2.17].
    Two scales: sigma1=0.5 px (fine), sigma2=1.5 px (medium).
    Strength alpha adapted from image contrast (Eq. 2.17).

    domain == remote:
        alpha = clip(1.0 + (0.4 - std) * 2.0, 0.3, 1.2)
        Slope halved, hard clip at 1.2 — prevents oversharpening on
        high-contrast land/cloud textures in satellite imagery.

    all others:
        alpha = clip(1.0 + (0.4 - std) * 4.0, 0.3, 2.5)  [Eq. 2.17]

    Suppressed (alpha *= 0.5) when Ringing Score > 8.0 [6].
    Note: deep_space uses sharpen_alpha=0.0 so this method is not called.
    """

    @staticmethod
    def apply(img: np.ndarray,
              alpha: Optional[float] = None,
              ring_threshold: float = 8.0,
              domain: str = ''):
        img = img.astype(np.float32)
        if alpha is None:
            std = float(np.std(img))
            if domain == DOMAIN_REMOTE:
                alpha = float(np.clip(1.0 + (0.4 - std) * 2.0, 0.3, 1.2))
            else:
                alpha = float(np.clip(1.0 + (0.4 - std) * 4.0, 0.3, 2.5))
        if SharpnessMetrics.ringing_score(img) > ring_threshold:
            alpha *= 0.5
        fine   = img - filters.gaussian(img, sigma=0.5)
        medium = img - filters.gaussian(img, sigma=1.5)
        out    = np.clip(img + alpha * fine + (alpha / 2.0) * medium, 0.0, 1.0)
        return out.astype(np.float32), float(alpha)


# ================================================================
# Exposure Adjustment
# ================================================================

def apply_exposure(img: np.ndarray, ev: float) -> np.ndarray:
    """
    Linear radiometric rescaling: out = clip(img * 2^ev, 0, 1).
    Does not alter PSF residuals or photometric ratios.
    ev > 0 -> brighter; ev < 0 -> darker.
    """
    return np.clip(img.astype(np.float32) * float(2.0 ** ev), 0.0, 1.0)


# ================================================================
# Restoration Engine
# ================================================================

class RestorationEngine:
    """
    Full pipeline: load -> PSF identify -> scale optimise (ringing + detail
    penalised) -> RL deconvolve (adaptive or fixed) -> sharpen.

    Colour strategy
    ---------------
    independent_channels=False (3-ch): BGR->CIE-Lab, deconvolve L only.
    independent_channels=True        : per-band RL (mono/multispectral).
    Grayscale and N!=3 always use independent mode.
    """

    def __init__(self, img_input, config: Optional[EngineConfig] = None):
        self.cfg = config or EngineConfig()
        if isinstance(img_input, str):
            self._load(img_input)
        else:
            self.img  = np.asarray(img_input, dtype=np.float32)
            self.n_ch = 1 if self.img.ndim == 2 else self.img.shape[2]

    # -- I/O ----------------------------------------------------------

    def _load(self, path: str) -> None:
        if path.lower().endswith(('.fits', '.fit', '.fts')):
            self._load_fits(path)
        else:
            self._load_opencv(path)

    def _load_opencv(self, path: str) -> None:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Cannot open: {path}")
        scale    = 65535.0 if raw.dtype == np.uint16 else 255.0
        self.img = raw.astype(np.float32) / scale
        self.n_ch = 1 if self.img.ndim == 2 else self.img.shape[2]
        if self.cfg.verbose:
            bits = 16 if raw.dtype == np.uint16 else 8
            print(f"[Load] {bits}-bit  ch={self.n_ch}  shape={raw.shape}")

    def _load_fits(self, path: str) -> None:
        if not _ASTROPY:
            raise ImportError("pip install astropy  (required for FITS)")
        with _fits.open(path) as hdul:
            data = hdul[0].data.astype(np.float32)
        lo = float(np.percentile(data, 0.1))
        hi = float(np.percentile(data, 99.9))
        data = np.clip((data - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        if data.ndim == 3:
            data = np.moveaxis(data, 0, -1)
        self.img  = data
        self.n_ch = 1 if self.img.ndim == 2 else self.img.shape[2]
        if self.cfg.verbose:
            print(f"[Load] FITS  ch={self.n_ch}  shape={self.img.shape}")

    # -- Adaptive defaults ---------------------------------------------

    def _psf_size(self) -> int:
        if self.cfg.psf_size is not None:
            return self.cfg.psf_size
        size = int(np.clip(max(self.img.shape[:2]) // 80, 11, 31))
        return size | 1

    def _gamma_bounds(self, model: str, sigma_est: float) -> tuple:
        """
        Data-driven gamma bounds centred on the spectral blur estimate.

        ЗМІНА 1: Planetary ceiling знижено з 6.0 до 4.5 px.
        ──────────────────────────────────────────────────────
          hi = min(s × 1.5, 4.5)

        Motivation: lunar/planetary PSF ≈ 1–2 px σ at 0.5"/px scale.
        4.5 px is the physical upper limit under poor seeing without a
        guide star. The previous 6.0 px ceiling allowed the optimiser to
        pick «soap» solutions amplified by sigma_est over-estimation
        (bright limbs/craters add HF power, reducing the measured blur).

        Other domains unchanged:
          microscopy  : min(s × 2.5, 4.0)   Airy radius 1–3 px
          remote      : min(s × 2.5, 6.0)   atmosphere + platform
          deep_space  : min(s × 3.0, 12.0)  wide seeing range
        """
        if self.cfg.gamma_bounds is not None:
            return self.cfg.gamma_bounds
        domain = self.cfg.domain
        s = sigma_est

        if model == 'motion':
            d = float(min(self.img.shape[:2]))
            return (1.0, float(np.clip(d / 80.0, 3.0, 25.0)))

        if model == 'scattering':
            return (0.05, 1.5)

        lo = max(0.3, s * 0.3)

        if domain == DOMAIN_MICRO:
            hi = min(s * 2.5, 4.0)
        elif domain == DOMAIN_REMOTE:
            hi = min(s * 2.5, 6.0)
        elif domain == DOMAIN_DEEP_SPACE:
            hi = min(s * 3.0, 12.0)
        elif domain == DOMAIN_PLANETARY:
            # ЗМІНА 1: tighter ceiling prevents over-broad PSF on craters
            hi = min(s * 1.5, 4.5)
        else:
            hi = min(s * 3.0, 10.0)

        hi = max(hi, lo + 0.5)   # ensure valid interval
        return (lo, hi)

    # -- Single-band pipeline -----------------------------------------

    def _process_channel(self, ch: np.ndarray, label: str = ''):
        cfg = self.cfg
        _v  = lambda s: print(f"  [{label}] {s}") if (cfg.verbose and label) \
                        else None

        # 0. Spectral blur-scale estimate (anchor for gamma bounds)
        sigma_est = estimate_blur_scale(ch)
        _v(f"sigma_est = {sigma_est:.3f} px")

        # 1. PSF model identification
        if cfg.psf_model == 'auto':
            model, det_angle, det_beta = PSFSelector.select(
                ch, domain=cfg.domain, verbose=cfg.verbose)
        else:
            model, det_angle, det_beta = cfg.psf_model, None, 3.5
            if cfg.verbose:
                print(f"  [PSF] override: {model}")

        beta  = cfg.moffat_beta if cfg.moffat_beta is not None else det_beta
        angle = (cfg.motion_angle if cfg.motion_angle is not None
                 else (det_angle or 0.0))

        # 2. Kernel size and data-driven search bounds
        size   = self._psf_size()
        bounds = self._gamma_bounds(model, sigma_est)

        # 3. PSF scale optimisation (ringing + detail penalised)
        opt   = GammaOptimizer(ch, model, size,
                                angle=angle, beta=beta, eps=cfg.rl_filter_eps,
                                domain=cfg.domain)
        gamma = opt.run(bounds, cfg.verbose)

        # 4. Build optimal kernel
        psf = PSFKernels.build(model, size, gamma, beta, angle)

        # 5. Richardson-Lucy deconvolution [3,4]
        min_gain = cfg.rl_min_gain_frac if cfg.rl_min_gain_frac is not None \
                   else 0.003
        # ЗМІНА 4: domain-aware ringing ceiling (6.0 for deep_space)
        ring_max = cfg.rl_ring_max if cfg.rl_ring_max is not None else 7.5

        if cfg.auto_iterations:
            restored, k_done = rl_adaptive(
                ch.astype(np.float64), psf.astype(np.float64),
                max_iter=cfg.rl_iterations,
                eps=cfg.rl_filter_eps,
                min_gain_frac=min_gain,
                ring_max=ring_max,
                verbose=cfg.verbose)
        else:
            restored = _rl_steps(ch.astype(np.float64),
                                  psf.astype(np.float64),
                                  cfg.rl_iterations, cfg.rl_filter_eps)
            k_done = cfg.rl_iterations

        meta = {'model': model, 'gamma': gamma, 'angle': angle,
                'beta': beta, 'psf_size': size, 'psf': psf,
                'sigma_est': sigma_est, 'rl_iters': k_done}
        return restored, meta

    # -- Colour strategies ---------------------------------------------

    def _process_lab(self):
        lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2Lab)
        L   = (lab[:, :, 0] / 100.0).astype(np.float32)
        if self.cfg.verbose:
            print("\n[Colour] CIE-Lab -> deconvolve L only")
        L_rest, meta     = self._process_channel(L, 'L')
        lab_out          = lab.copy()
        lab_out[:, :, 0] = np.clip(L_rest * 100.0, 0.0, 100.0)
        out = cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)
        return np.clip(out, 0.0, 1.0).astype(np.float32), meta

    def _process_independent(self):
        if self.img.ndim == 2:
            return self._process_channel(self.img, 'ch0')
        results, metas = [], []
        for i in range(self.n_ch):
            if self.cfg.verbose:
                print(f"\n[Band {i}]")
            r, m = self._process_channel(self.img[:, :, i], f'B{i}')
            results.append(r)
            metas.append(m)
        return np.stack(results, axis=-1), metas[0]

    # -- Public API ---------------------------------------------------

    def process(self) -> dict:
        """
        Returns dict with keys:
          result    : final image (float32, [0,1])
          original  : input
          restored  : after RL, before sharpening
          alpha     : sharpening strength
          metrics   : Tenengrad, Laplacian variance, Ringing score
          + PSF metadata (model, gamma, beta, angle, sigma_est, rl_iters, ...)
        """
        cfg = self.cfg
        if cfg.verbose:
            print("\n" + "=" * 60)
            print(f"  RestorationEngine  domain={cfg.domain}  "
                  f"auto_iter={cfg.auto_iterations}")
            print("=" * 60)

        use_ind = (self.img.ndim == 2 or self.n_ch != 3
                   or cfg.independent_channels)
        if use_ind:
            restored, meta = self._process_independent()
        else:
            restored, meta = self._process_lab()

        # Post-deconvolution sharpening
        if cfg.sharpen_alpha == 0.0:
            final, alpha_used = restored, 0.0
        else:
            if cfg.verbose:
                print("\n[Sharpen] Multiscale unsharp masking")
            final, alpha_used = AdaptiveSharpen.apply(
                restored, cfg.sharpen_alpha, domain=cfg.domain)

        # Metrics on luminance plane
        lum_o = _luminance(self.img)
        lum_f = _luminance(final)
        t0  = SharpnessMetrics.tenengrad(lum_o)
        t1  = SharpnessMetrics.tenengrad(lum_f)
        lv0 = SharpnessMetrics.laplacian_variance(lum_o)
        lv1 = SharpnessMetrics.laplacian_variance(lum_f)
        rng = SharpnessMetrics.ringing_score(lum_f)

        metrics = {
            'tenengrad_before':     t0,
            'tenengrad_after':      t1,
            'tenengrad_gain_pct':   (t1 - t0) / (t0 + 1e-9) * 100,
            'laplacian_var_before': lv0,
            'laplacian_var_after':  lv1,
            'ringing_score':        rng,
        }

        if cfg.verbose:
            _print_report(meta, alpha_used, metrics, cfg)

        return {'result': final, 'original': self.img,
                'restored': restored, 'alpha': alpha_used,
                'metrics': metrics, **meta}


# -- Internal helpers --------------------------------------------------

def _luminance(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:      return img
    if img.shape[2] == 1:  return img[:, :, 0]
    if img.shape[2] == 3:
        return (0.114 * img[:, :, 0] + 0.587 * img[:, :, 1]
                + 0.299 * img[:, :, 2]).astype(np.float32)
    return img[:, :, 0]


def _print_report(meta: dict, alpha: float,
                  m: dict, cfg: EngineConfig) -> None:
    rs          = m['ringing_score']
    warn        = '  WARNING: reduce iterations' if rs > 8 else '  OK'
    ring_max_v  = cfg.rl_ring_max if cfg.rl_ring_max is not None else 7.5
    print("\n" + "=" * 60)
    print(f"  domain      : {cfg.domain}")
    print(f"  PSF model   : {meta.get('model','?')}  "
          f"gamma*={meta.get('gamma',0):.4f}  "
          f"sigma_est={meta.get('sigma_est',0):.3f} px")
    if meta.get('model') == 'moffat':
        print(f"  Moffat beta : {meta.get('beta',3.5):.1f}")
    if meta.get('model') == 'motion':
        print(f"  Blur angle  : {meta.get('angle',0):.1f} deg")
    print(f"  RL iters    : {meta.get('rl_iters','?')}  "
          f"(auto={cfg.auto_iterations}  ring_max={ring_max_v:.1f})")
    print(f"  Sharpen a   : {alpha:.3f}")
    print(f"  Tenengrad   : {m['tenengrad_before']:.3f} -> "
          f"{m['tenengrad_after']:.3f}  "
          f"(+{m['tenengrad_gain_pct']:.1f}%)")
    print(f"  Lapl.var.   : {m['laplacian_var_before']:.3f} -> "
          f"{m['laplacian_var_after']:.3f}")
    print(f"  Ringing     : {rs:.2f}{warn}")
    print("=" * 60)