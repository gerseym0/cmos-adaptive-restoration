"""
GUI - Adaptive PSF Deconvolution for Scientific CMOS Imaging
Run:  python diploma_gui.py
"""

from __future__ import annotations
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from diploma_engine import (
    EngineConfig, RestorationEngine, domain_config, apply_exposure,
    DOMAIN_KEYS, DOMAIN_DISPLAY,
)

# ── Palette ──────────────────────────────────────────────────────────────────
BG      = '#12141c'
PANEL   = '#1c1f2e'
BORDER  = '#282c3e'
ACCENT  = '#5b8af0'
TEXT    = '#dde0f0'
DIM     = '#6b7090'
GREEN   = '#4caf7d'
ORANGE  = '#f0a040'
RED     = '#e05555'
ENTRY   = '#22253a'

PSF_MAP = {
    'Авто (ідентифікація)':            'auto',
    'Gaussian — атмосферний blur':     'gaussian',
    'Cauchy — сильна турбулентність':  'cauchy',
    'Airy disk — дифракційна межа':    'airy',
    'Moffat — atmospheric seeing':     'moffat',
    'Motion blur — розмиття руху':     'motion',
    'Scattering — розсіювання BSI':    'scattering',
}
PSF_LABELS = list(PSF_MAP.keys())


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_rgb(img: np.ndarray) -> np.ndarray:
    """float32 [0,1] any layout -> uint8 RGB for matplotlib."""
    if img.ndim == 2:
        return (img * 255).clip(0, 255).astype(np.uint8)
    if img.shape[2] == 1:
        return (img[:, :, 0] * 255).clip(0, 255).astype(np.uint8)
    if img.shape[2] == 3:
        return cv2.cvtColor((img * 255).clip(0, 255).astype(np.uint8),
                            cv2.COLOR_BGR2RGB)
    return (img[:, :, :3] * 255).clip(0, 255).astype(np.uint8)


def is_gray(img: np.ndarray) -> bool:
    return img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)


# ── Reusable widget factories ──────────────────────────────────────────────────

def make_label(parent, text, color=DIM, font=('Segoe UI', 8), **kw):
    return tk.Label(parent, text=text, bg=PANEL, fg=color, font=font, **kw)


def make_sep(parent):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill='x', pady=6)


def make_btn(parent, text, cmd, bg=ACCENT, fg=TEXT,
             font=('Segoe UI', 9), **kw):
    return tk.Button(parent, text=text, command=cmd,
                     bg=bg, fg=fg, relief='flat',
                     activebackground=BORDER, activeforeground=fg,
                     font=font, cursor='hand2', padx=8, pady=5, **kw)


# ── Application ────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Adaptive PSF Deconvolution  |  CMOS Scientific Imaging")
        self.configure(bg=BG)
        self.minsize(1100, 640)
        self.geometry('1260x720')

        self._output: dict | None = None
        self._file_path: str | None = None
        self._preview: np.ndarray | None = None

        self._ev_var   = tk.DoubleVar(value=0.0)
        self._auto_var = tk.BooleanVar(value=True)
        self._ev_var.trace_add('write', lambda *_: self._redraw())

        self._build_styles()
        self._build_ui()

    # ── Styles ────────────────────────────────────────────────────────────────

    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use('clam')
        for name, bg_ in [('Panel.TFrame', PANEL), ('BG.TFrame', BG)]:
            s.configure(name, background=bg_)
        s.configure('Dark.TCombobox',
                    fieldbackground=ENTRY, background=ENTRY,
                    foreground=TEXT, arrowcolor=ACCENT,
                    selectbackground=ACCENT)
        s.map('Dark.TCombobox',
              fieldbackground=[('readonly', ENTRY)],
              foreground=[('readonly', TEXT)])
        s.configure('T.Horizontal.TScale',
                    background=PANEL, troughcolor=BORDER, sliderthickness=13)
        s.configure('Prog.Horizontal.TProgressbar',
                    troughcolor=BORDER, background=ACCENT,
                    borderwidth=0, thickness=4)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar
        bar = tk.Frame(self, bg=BG, height=40)
        bar.pack(fill='x')
        tk.Label(bar, text="PSF Deconvolution Engine",
                 bg=BG, fg=ACCENT, font=('Segoe UI', 12, 'bold')
                 ).pack(side='left', padx=16, pady=8)
        tk.Label(bar, text="Адаптивна фільтрація зображень для CMOS-сенсорів",
                 bg=BG, fg=DIM, font=('Segoe UI', 9)
                 ).pack(side='left', pady=8)
        tk.Frame(self, bg=BORDER, height=1).pack(fill='x')

        # Body
        body = tk.Frame(self, bg=BG)
        body.pack(fill='both', expand=True, padx=10, pady=10)

        self._build_panel(body)   # left
        self._build_canvas(body)  # right

    def _build_panel(self, parent):
        panel = tk.Frame(parent, bg=PANEL, width=272)
        panel.pack(side='left', fill='y', padx=(0, 10))
        panel.pack_propagate(False)
        P = panel

        def row(text, color=DIM):
            make_label(P, text, color=color).pack(
                anchor='w', padx=12, pady=(6, 0))

        # File
        make_sep(P)
        row("Вхідний файл", TEXT)
        fr = tk.Frame(P, bg=PANEL)
        fr.pack(fill='x', padx=10, pady=4)
        make_btn(fr, "Відкрити", self._open_file,
                 bg=ENTRY, fg=TEXT, font=('Segoe UI', 8)
                 ).pack(side='left')
        self._lbl_file = tk.Label(fr, text="не вибрано", bg=PANEL, fg=DIM,
                                   font=('Segoe UI', 8), anchor='w',
                                   wraplength=155)
        self._lbl_file.pack(side='left', padx=6)

        # Domain
        make_sep(P)
        row("Тематика", TEXT)
        self._domain_var = tk.StringVar(value=DOMAIN_DISPLAY[0])
        cb = ttk.Combobox(P, textvariable=self._domain_var,
                          values=DOMAIN_DISPLAY, state='readonly',
                          style='Dark.TCombobox', width=30)
        cb.pack(padx=10, pady=4, fill='x')
        cb.bind('<<ComboboxSelected>>', self._on_domain)
        self._lbl_hint = tk.Label(P, text="", bg=PANEL, fg=DIM,
                                   font=('Segoe UI', 7, 'italic'),
                                   wraplength=248, justify='left')
        self._lbl_hint.pack(anchor='w', padx=12)

        # PSF model
        make_sep(P)
        row("Модель PSF", TEXT)
        self._psf_var = tk.StringVar(value=PSF_LABELS[0])
        cb_psf = ttk.Combobox(P, textvariable=self._psf_var,
                               values=PSF_LABELS, state='readonly',
                               style='Dark.TCombobox', width=30)
        cb_psf.pack(padx=10, pady=4, fill='x')

        # Iterations
        make_sep(P)
        row("Ітерації RL", TEXT)

        fr_auto = tk.Frame(P, bg=PANEL)
        fr_auto.pack(fill='x', padx=10, pady=(2, 0))
        tk.Checkbutton(fr_auto, text="Авто-зупинка (за збіжністю)",
                        variable=self._auto_var, bg=PANEL, fg=TEXT,
                        selectcolor=ENTRY, activebackground=PANEL,
                        font=('Segoe UI', 8),
                        command=self._on_auto_toggle
                        ).pack(side='left')

        self._fr_iter = tk.Frame(P, bg=PANEL)
        self._fr_iter.pack(fill='x', padx=10, pady=2)
        self._iter_var = tk.IntVar(value=30)
        sc = ttk.Scale(self._fr_iter, from_=5, to=80, orient='horizontal',
                       variable=self._iter_var, style='T.Horizontal.TScale',
                       length=200)
        sc.pack(side='left')
        self._lbl_iter = tk.Label(self._fr_iter, text="30",
                                   bg=PANEL, fg=ACCENT,
                                   font=('Segoe UI', 9, 'bold'), width=3)
        self._lbl_iter.pack(side='left', padx=4)
        self._iter_var.trace_add('write',
            lambda *_: self._lbl_iter.config(
                text=str(int(self._iter_var.get()))))
        self._lbl_iter_cap = make_label(
            P, "максимум (при авто) або точне (при ручному)")
        self._lbl_iter_cap.pack(anchor='w', padx=12)

        # Sharpening
        make_sep(P)
        row("Загострення  (0 = вимкнено)", TEXT)
        fr_sh = tk.Frame(P, bg=PANEL)
        fr_sh.pack(fill='x', padx=10, pady=2)
        self._sh_var = tk.DoubleVar(value=0.0)
        sc_sh = ttk.Scale(fr_sh, from_=0.0, to=3.0, orient='horizontal',
                          variable=self._sh_var, style='T.Horizontal.TScale',
                          length=200)
        sc_sh.pack(side='left')
        self._lbl_sh = tk.Label(fr_sh, text="0.0",
                                 bg=PANEL, fg=TEXT, font=('Segoe UI', 8),
                                 width=4)
        self._lbl_sh.pack(side='left', padx=4)
        self._sh_var.trace_add('write',
            lambda *_: self._lbl_sh.config(
                text=f"{self._sh_var.get():.1f}"))

        # Exposure
        make_sep(P)
        row("Експозиція результату  (EV)", TEXT)
        fr_ev = tk.Frame(P, bg=PANEL)
        fr_ev.pack(fill='x', padx=10, pady=2)
        sc_ev = ttk.Scale(fr_ev, from_=-3.0, to=3.0, orient='horizontal',
                          variable=self._ev_var, style='T.Horizontal.TScale',
                          length=180)
        sc_ev.pack(side='left')
        self._lbl_ev = tk.Label(fr_ev, text="+0.0 EV",
                                 bg=PANEL, fg=ACCENT,
                                 font=('Segoe UI', 8, 'bold'), width=7)
        self._lbl_ev.pack(side='left', padx=2)
        self._ev_var.trace_add('write',
            lambda *_: self._lbl_ev.config(
                text=f"{self._ev_var.get():+.1f} EV"))
        tk.Button(fr_ev, text="0", bg=BORDER, fg=DIM, relief='flat',
                  font=('Segoe UI', 8), padx=4,
                  command=lambda: self._ev_var.set(0.0)
                  ).pack(side='left', padx=2)

        # Run button
        make_sep(P)
        self._btn_run = make_btn(
            P, "Обробити", self._run,
            bg=ACCENT, font=('Segoe UI', 10, 'bold'))
        self._btn_run.pack(fill='x', padx=10, pady=(0, 4))

        self._prog = ttk.Progressbar(P, mode='indeterminate', length=250,
                                      style='Prog.Horizontal.TProgressbar')
        self._prog.pack(padx=10)

        self._lbl_status = tk.Label(P, text="", bg=PANEL, fg=DIM,
                                     font=('Segoe UI', 8, 'italic'))
        self._lbl_status.pack(pady=2)

        # Metrics
        make_sep(P)
        self._lbl_metrics = tk.Label(
            P, text="немає результату",
            bg=PANEL, fg=DIM, font=('Courier New', 8),
            justify='left', anchor='w')
        self._lbl_metrics.pack(anchor='w', padx=12, pady=2)

        # Save
        make_sep(P)
        self._btn_save = make_btn(P, "Зберегти результат",
                                   self._save, bg='#2e6e50')
        self._btn_save.pack(fill='x', padx=10, pady=(0, 8))
        self._btn_save.config(state='disabled')

        self._on_domain()
        self._on_auto_toggle()

    def _build_canvas(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.pack(side='left', fill='both', expand=True)

        self._fig = Figure(facecolor=BG, tight_layout=True)
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill='both', expand=True)
        self._draw_placeholder()

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_domain(self, *_):
        idx = (DOMAIN_DISPLAY.index(self._domain_var.get())
               if self._domain_var.get() in DOMAIN_DISPLAY else 0)
        domain = DOMAIN_KEYS[idx]

        # ЗМІНА А: замінено окремі ключі 'stellar'/'extended' на 'deep_space'
        hints = {
            'deep_space': (
                "Moffat PSF з адаптивним beta [Moffat 1969]; загострення "
                "вимкнено (збереження фотометричного потоку); посилений штраф "
                "за ringing (λ=0.40); незалежні канали. Для зоряних полів, "
                "туманностей та галактик."
            ),
            'planetary': (
                "Авто PSF; вужчий діапазон пошуку gamma (≤6 px) — запобігає "
                "«милу» на кратерах; знижений поріг збіжності RL (0.001) для "
                "більшої кількості ітерацій; помірне загострення."
            ),
            'microscopy': (
                "Airy disk (дифракційна межа [Born&Wolf]); вузькі межі gamma "
                "(<4 px) — запобігання кільцям та втраті мікроконтрасту; "
                "50 ітерацій; незалежні канали."
            ),
            'remote': (
                "Авто PSF; знижений поріг motion blur (платформна вібрація); "
                "gamma ≤ 6 px; пом'якшене загострення (множник 2.0, clip 1.2) "
                "для запобігання перешарпу на контрастних текстурах суші."
            ),
        }
        self._lbl_hint.config(text=hints.get(domain, ""))
        cfg = domain_config(domain)
        self._iter_var.set(cfg.rl_iterations)
        sh = cfg.sharpen_alpha if cfg.sharpen_alpha is not None else 0.0
        self._sh_var.set(round(sh, 1))

    def _on_auto_toggle(self):
        if self._auto_var.get():
            self._lbl_iter_cap.config(
                text="максимум ітерацій (зупинка за збіжністю)",
                fg=DIM)
        else:
            self._lbl_iter_cap.config(
                text="точна кількість ітерацій (без авто-зупинки)",
                fg=ORANGE)

    # ── File ──────────────────────────────────────────────────────────────────

    def _open_file(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Зображення", "*.tif *.tiff *.png *.jpg *.jpeg "
                               "*.fit *.fits *.fts"),
                ("All files", "*.*"),
            ])
        if not path:
            return
        self._file_path = path
        self._lbl_file.config(text=Path(path).name, fg=TEXT)
        try:
            raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if raw is not None:
                scale = 65535.0 if raw.dtype == np.uint16 else 255.0
                self._preview = raw.astype(np.float32) / scale
                self._output  = None
                self._draw_single(self._preview, "Оригінал")
        except Exception:
            pass

    # ── Processing ────────────────────────────────────────────────────────────

    def _run(self):
        if not self._file_path:
            messagebox.showwarning("Файл не вибрано",
                                   "Спочатку відкрийте зображення.")
            return

        idx    = (DOMAIN_DISPLAY.index(self._domain_var.get())
                  if self._domain_var.get() in DOMAIN_DISPLAY else 0)
        domain = DOMAIN_KEYS[idx]
        psf_k  = PSF_MAP.get(self._psf_var.get(), 'auto')
        sh_raw = self._sh_var.get()
        sha    = None if sh_raw < 0.05 else float(sh_raw)

        cfg = domain_config(
            domain,
            psf_model=psf_k,
            rl_iterations=int(self._iter_var.get()),
            auto_iterations=self._auto_var.get(),
            sharpen_alpha=sha,
            verbose=True,
        )

        self._btn_run.config(state='disabled')
        self._btn_save.config(state='disabled')
        self._prog.start(12)
        self._lbl_status.config(text="обробка...", fg=ORANGE)

        def worker():
            try:
                eng = RestorationEngine(self._file_path, cfg)
                out = eng.process()
                self.after(0, lambda: self._done(out))
            except Exception as exc:
                self.after(0, lambda: self._error(str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _done(self, output: dict):
        self._output = output
        self._prog.stop()
        self._btn_run.config(state='normal')
        self._btn_save.config(state='normal')
        m = output['metrics']
        rs = m['ringing_score']
        k  = output.get('rl_iters', '?')
        status = (f"готово  k={k}  R={rs:.1f}"
                  + ("  ⚠" if rs > 8 else "  OK"))
        self._lbl_status.config(
            text=status, fg=RED if rs > 8 else GREEN)
        self._update_metrics(output)
        self._redraw()

    def _error(self, msg: str):
        self._prog.stop()
        self._btn_run.config(state='normal')
        self._lbl_status.config(text="помилка", fg=RED)
        messagebox.showerror("Помилка", msg)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _update_metrics(self, out: dict):
        m   = out['metrics']
        rs  = m['ringing_score']
        s_est = out.get('sigma_est', 0)
        gamma = out.get('gamma', 0)
        k   = out.get('rl_iters', '?')
        col = RED if rs > 8 else GREEN
        txt = (
            f"PSF     {out.get('model','?')}  g={gamma:.3f}\n"
            f"sigma   {s_est:.2f} px\n"
            f"iters   {k}\n"
            f"Tenengrad  +{m['tenengrad_gain_pct']:.1f}%\n"
            f"Ringing    {rs:.2f}"
            + ("  RING!" if rs > 8 else "")
        )
        self._lbl_metrics.config(text=txt, fg=col)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _redraw(self):
        if self._output is None:
            if self._preview is not None:
                self._draw_single(self._preview, "Оригінал")
            return
        ev   = self._ev_var.get()
        orig = self._output['original']
        final = apply_exposure(self._output['result'], ev)
        self._draw_compare(orig, final, self._output)

    def _draw_placeholder(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor(BG)
        ax.text(0.5, 0.5,
                "Відкрийте файл та натисніть «Обробити»",
                ha='center', va='center', color=DIM,
                fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        self._canvas.draw()

    def _draw_single(self, img: np.ndarray, title: str = ""):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor(BG)
        ax.imshow(to_rgb(img), cmap='gray' if is_gray(img) else None,
                  interpolation='nearest')
        ax.set_title(title, color=TEXT, fontsize=10, pad=6)
        ax.axis('off')
        self._fig.tight_layout(pad=0.5)
        self._canvas.draw()

    def _draw_compare(self, orig: np.ndarray, final: np.ndarray,
                       out: dict):
        self._fig.clear()
        self._fig.patch.set_facecolor(BG)

        m     = out['metrics']
        model = out.get('model', '?')
        gamma = out.get('gamma', 0.0)
        k     = out.get('rl_iters', '?')
        rs    = m['ringing_score']
        ev    = self._ev_var.get()

        # Left: original
        ax1 = self._fig.add_axes([0.01, 0.06, 0.48, 0.88])
        ax1.set_facecolor(BG)
        ax1.imshow(to_rgb(orig), cmap='gray' if is_gray(orig) else None,
                   interpolation='nearest')
        ax1.set_title("Оригінал", color=TEXT, fontsize=10, pad=6)
        t0_s = f"T = {m['tenengrad_before']:.4f}"
        ax1.set_xlabel(t0_s, color=DIM, fontsize=8, labelpad=4)
        ax1.tick_params(left=False, bottom=False,
                        labelleft=False, labelbottom=False)

        # Right: result
        ax2 = self._fig.add_axes([0.51, 0.06, 0.48, 0.88])
        ax2.set_facecolor(BG)
        ax2.imshow(to_rgb(final), cmap='gray' if is_gray(final) else None,
                   interpolation='nearest')
        ev_s = f"  EV {ev:+.1f}" if ev != 0.0 else ""
        ax2.set_title(f"Результат  [{model} g={gamma:.3f} k={k}]{ev_s}",
                      color=TEXT, fontsize=10, pad=6)
        t1_s = (f"T = {m['tenengrad_after']:.4f}  "
                f"(+{m['tenengrad_gain_pct']:.1f}%)  "
                f"R={rs:.1f}" + ("  ⚠" if rs > 8 else ""))
        ring_col = '#e05555' if rs > 8 else '#4caf7d'
        ax2.set_xlabel(t1_s, color=ring_col, fontsize=8, labelpad=4)
        ax2.tick_params(left=False, bottom=False,
                        labelleft=False, labelbottom=False)

        # Thin divider line
        self._fig.add_artist(
            matplotlib.lines.Line2D([0.503, 0.503], [0.02, 0.98],
                                     color=BORDER, linewidth=1,
                                     transform=self._fig.transFigure))

        self._canvas.draw()

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save(self):
        if self._output is None:
            return
        ev    = self._ev_var.get()
        final = apply_exposure(self._output['result'], ev)

        path = filedialog.asksaveasfilename(
            defaultextension='.tiff',
            filetypes=[
                ("16-bit TIFF (наукове)", "*.tiff"),
                ("PNG 8-bit",             "*.png"),
                ("JPEG 8-bit",            "*.jpg"),
            ])
        if not path:
            return

        p = path.lower()
        if p.endswith(('.tif', '.tiff')):
            cv2.imwrite(path, (final * 65535).astype(np.uint16))
        elif p.endswith('.png'):
            cv2.imwrite(path, (final * 255).clip(0, 255).astype(np.uint8))
        else:
            cv2.imwrite(path, (final * 255).clip(0, 255).astype(np.uint8),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

        messagebox.showinfo("Збережено", f"Файл збережено:\n{path}")


# ── Entry point ───────────────────────────────────────────────────────────────

import matplotlib.lines   # ensure available for Line2D

if __name__ == '__main__':
    App().mainloop()
