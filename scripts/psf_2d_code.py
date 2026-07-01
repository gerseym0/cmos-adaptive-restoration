"""
Рисунок 2.Y — Двовимірна візуалізація ядер PSF (heatmap)
6 моделей на одному полотні, log-scale кольорова шкала
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import j1

# ── PSF ядра (2D, розмір SIZExSIZE) ──────────────────────────────────────
SIZE = 101   # непарне → центр у (50,50)
ax_1d = np.arange(-SIZE//2, SIZE//2 + 1, dtype=np.float64)
XX, YY = np.meshgrid(ax_1d, ax_1d)
R2 = XX**2 + YY**2
R  = np.sqrt(R2)

def make_gaussian(sigma=2.5):
    k = np.exp(-R2 / (2*sigma**2))
    return k / k.max()

def make_cauchy(gamma=2.5):
    k = (1 + R2/gamma**2)**(-1.5)
    return k / k.max()

def make_airy(radius=3.0):
    with np.errstate(invalid='ignore', divide='ignore'):
        arg = np.pi * R / radius
        k   = np.where(R == 0, 1.0, (2*j1(arg)/arg)**2)
    return k / k.max()

def make_moffat(alpha=2.5, beta=3.5):
    k = (1 + R2/alpha**2)**(-beta)
    return k / k.max()

def make_motion(length=8.0, angle_deg=30.0):
    kernel = np.zeros_like(R)
    cx = SIZE//2
    cos_a = np.cos(np.deg2rad(angle_deg))
    sin_a = np.sin(np.deg2rad(angle_deg))
    half  = length / 2.0
    for t in np.linspace(-half, half, max(int(6*length), 20)):
        xi = int(round(cx + t * cos_a))
        yi = int(round(cx + t * sin_a))
        if 0 <= xi < SIZE and 0 <= yi < SIZE:
            kernel[yi, xi] = 1.0
    # Gaussian smooth for visualization
    from scipy.ndimage import gaussian_filter
    kernel = gaussian_filter(kernel, sigma=0.6)
    return kernel / kernel.max()

def make_scattering(k_val=0.18):
    """Виправлена модель: k²/(2π)·exp(-k·r), нормована на максимум"""
    kv = np.exp(-k_val * R)
    return kv / kv.max()

# ── дані для шести панелей ─────────────────────────────────────────────────
panels = [
    ('Гаусова\n(Gaussian, σ=2.5 пкс)',   make_gaussian()),
    ('Коші\n(Cauchy, γ=2.5 пкс)',         make_cauchy()),
    ('Ейрі\n(Airy disk, R=3.0 пкс)',      make_airy()),
    ('Моффат\n(Moffat, α=2.5, β=3.5)',    make_moffat()),
    ('Розсіювання\n(Scattering, k=0.18)', make_scattering()),
    ('Розмиття руху\n(Motion, L=8, θ=30°)', make_motion()),
]

# ── обрізка для відображення (показуємо центральну область) ───────────────
crop = 30   # ±30 px від центру
c    = SIZE // 2
sl   = slice(c - crop, c + crop + 1)

# ── побудова ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 9))
fig.patch.set_facecolor('white')
fig.suptitle('Двовимірні ядра функцій розсіювання точки (PSF)\n'
             '(логарифмічна кольорова шкала, нормовано до максимуму)',
             fontsize=14, y=1.01)

VMIN = 1e-4   # мінімум логарифмічного діапазону

for ax, (title, kernel) in zip(axes.flat, panels):
    im = ax.imshow(
        np.maximum(kernel[sl, sl], VMIN),
        norm=LogNorm(vmin=VMIN, vmax=1.0),
        cmap='inferno',
        origin='lower',
        extent=[-crop, crop, -crop, crop],
        interpolation='bilinear'
    )

    # ── контурні лінії на рівнях 50%, 10%, 1% ──────────────────────────
    levels = [0.01, 0.10, 0.50]
    cs = ax.contour(
        np.maximum(kernel[sl, sl], VMIN),
        levels=levels,
        colors=['#aaaaaa', '#dddddd', 'white'],
        linewidths=[0.7, 0.8, 1.0],
        extent=[-crop, crop, -crop, crop],
        origin='lower'
    )
    labels = {0.01: '1%', 0.10: '10%', 0.50: 'HWHM 50%'}
    fmt    = {lv: labels[lv] for lv in levels}
    ax.clabel(cs, fmt=fmt, fontsize=8, inline=True)

    # ── кольорова шкала ────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Відн. інтенс.', fontsize=8)

    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel('Δx, пікселів', fontsize=9)
    ax.set_ylabel('Δy, пікселів', fontsize=9)
    ax.tick_params(labelsize=8)

    # ── перехрестя в центрі ────────────────────────────────────────────
    ax.axhline(0, color='white', linewidth=0.4, alpha=0.4)
    ax.axvline(0, color='white', linewidth=0.4, alpha=0.4)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('C:\\Users\\ibatu\\OneDrive\\Desktop\\diploma code\\psf_2d_heatmaps.png',
            dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: psf_2d_heatmaps.png")
