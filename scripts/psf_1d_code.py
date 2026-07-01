"""
Рисунок 2.X — Одновимірні перерізи моделей PSF (log scale)
Усі шість моделей нормовано: max = 1 (пік у центрі)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.special import j1

# ── параметри ──────────────────────────────────────────────────────────────
N       = 500          # точок
r_max   = 12.0         # пікселів від центру
r       = np.linspace(0, r_max, N)
EPS     = 1e-12        # захист від log(0)

# ── 6 моделей PSF ──────────────────────────────────────────────────────────
def gaussian(r, sigma=1.5):
    return np.exp(-r**2 / (2*sigma**2))

def cauchy(r, gamma=1.5):
    return (1 + r**2/gamma**2)**(-1.5)

def airy(r, R=1.5):
    with np.errstate(invalid='ignore', divide='ignore'):
        arg = np.pi * r / R
        k   = np.where(r == 0, 1.0, (2*j1(arg)/arg)**2)
    return k

def moffat(r, alpha=1.5, beta=3.5):
    return (1 + r**2/alpha**2)**(-beta)

def motion_1d(r, L=3.0):
    """
    1D перетин motion-PSF: рівномірний прямокутник шириною L,
    потім нуль. Нормовано на пік.
    """
    return np.where(r <= L/2, 1.0, EPS)

def scattering(r, k=0.8):
    """Правильна (виправлена) нормована форма: exp(-k*r)"""
    return np.exp(-k * r)

# ── побудова ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

styles = [
    ('Гаусова (Gaussian)',    gaussian(r),       '#1f77b4', '-',  2.0),
    ('Коші (Cauchy)',         cauchy(r),          '#d62728', '--', 2.0),
    ('Ейрі (Airy disk)',      airy(r),            '#2ca02c', '-.',  2.0),
    ('Моффат (Moffat β=3.5)',  moffat(r),         '#ff7f0e', ':',  2.2),
    ('Розсіювання (Scatt.)',  scattering(r),      '#9467bd', '--', 1.8),
    ('Розмиття руху (Motion)', motion_1d(r),      '#8c564b', '-',  1.8),
]

for label, vals, color, ls, lw in styles:
    vals_norm = vals / vals[0]   # нормуємо: максимум = 1
    ax.semilogy(r, np.maximum(vals_norm, EPS),
                label=label, color=color, linestyle=ls, linewidth=lw)

# ── оформлення ─────────────────────────────────────────────────────────────
ax.set_xlim(0, r_max)
ax.set_ylim(1e-4, 2.0)
ax.set_xlabel('Радіальна відстань від центру, пікселів', fontsize=13)
ax.set_ylabel('Відносна інтенсивність (нормована, лог. шкала)', fontsize=13)
ax.set_title('Одновимірні перерізи моделей PSF\n(нормовано до одиничного максимуму)', fontsize=14)

# позначки на осі Y: десяткові порядки
ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
ax.grid(True, which='major', linestyle='-',  linewidth=0.5, alpha=0.6, color='#cccccc')
ax.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.3, color='#dddddd')

# вертикальна лінія на рівні 0.5 (HWHM)
ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)
ax.text(r_max*0.98, 0.52, 'HWHM = 0.5', ha='right', fontsize=10, color='gray')

leg = ax.legend(fontsize=11, framealpha=0.9, edgecolor='#cccccc',
                loc='upper right')

# анотація важкохвостих профілів
ax.annotate('Важкі хвости\n(Коші, Моффат)',
            xy=(8.5, cauchy(np.array([8.5]))[0] / cauchy(np.array([0.0]))[0]),
            xytext=(6.0, 2e-3),
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2),
            fontsize=10, color='#d62728')
ax.annotate('',
            xy=(8.5, moffat(np.array([8.5]))[0] / moffat(np.array([0.0]))[0]),
            xytext=(6.2, 2.2e-3),
            arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.2))

plt.tight_layout()
plt.savefig('C:\\Users\\ibatu\\OneDrive\\Desktop\\diploma code\\psf_1d_profiles.png',
            dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: psf_1d_profiles.png")
