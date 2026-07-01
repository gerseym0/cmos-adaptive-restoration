# Adaptive Image Restoration for Scientific CMOS Sensors

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![SciPy](https://img.shields.io/badge/SciPy-1.11+-lightgrey.svg)

This repository contains an automated software system for adaptive filtering and image restoration, developed specifically for processing scientific images acquired via CMOS sensors. The system identifies the nature of optical blur dynamically and applies Richardson-Lucy deconvolution with specialized Point Spread Function (PSF) models.

This project was developed as part of a Bachelor's Thesis in Computer Engineering at Taras Shevchenko National University of Kyiv.

## 🌟 Key Features

* **Intelligent PSF Identification:** Automatically selects the best-fitting optical model based on domain parameters and statistical analysis (gradient kurtosis and directional ratio).
* **Specialized Optical Models:** Supports 6 analytical models[cite: 1]:
  * *Gaussian* (Atmospheric blur)
  * *Moffat / Cauchy* (Heavy-tailed astronomical seeing / turbulence)
  * *Airy Disk* (Diffraction-limited optical systems, e.g., microscopy)
  * *Scattering / Motion Blur*
* **16-bit & FITS Support:** Processes high-dynamic-range scientific data (TIFF, FITS) without data loss, common in astrophotography and biomedical imaging[cite: 1].
* **Interactive GUI:** Built-in Tkinter graphical interface for visual parameter tuning, domain selection (Deep Space, Planetary, Microscopy, Remote Sensing), and real-time preview[cite: 1].
* **Benchmarking Suite:** Automated tool to compare performance against standard methods (Wiener filters, Gaussian unsharp masks) generating Excel reports and metric plots[cite: 1].

## 📁 Repository Structure

* `app/` — Core application files:
  * `diploma_engine.py` (RL deconvolution, PSF math models, image metrics)
  * `diploma_gui.py` (Graphical User Interface)
* `scripts/` — Utility and benchmarking scripts:
  * `benchmark_comparison.py` (Automated testing & metrics generator)
  * `psf_1d_code.py` (Generator for 1D PSF profile visualizations)
  * `psf_2d_code.py` (Generator for 2D PSF heatmaps)
* `data/` — Directory for test images.
* `docs/` — Documentation:
  * `Diploma_Thesis.pdf` (Full text of the bachelor's thesis)
* `requirements.txt` — Python dependencies.

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/gerseym0/cmos-adaptive-restoration.git](https://github.com/gerseym0/cmos-adaptive-restoration.git)
   cd cmos-adaptive-restoration
