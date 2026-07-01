# Adaptive Image Restoration for Scientific CMOS Sensors

This repository contains an automated software system for adaptive filtering and image restoration. It is specifically designed for processing scientific images acquired via CMOS sensors in domains such as deep space astrophotography, planetary imaging, remote sensing, and fluorescence microscopy.

The system dynamically identifies the nature of optical blur and applies Richardson-Lucy deconvolution optimized with specialized Point Spread Function (PSF) models to recover image sharpness without introducing ringing artifacts.

This project was developed as a Bachelor's Thesis in Computer Engineering at Taras Shevchenko National University of Kyiv.

## Key Features

* **Intelligent PSF Identification:** Automatically selects the best-fitting optical model based on domain parameters, utilizing Gradient Kurtosis and Directional Ratio statistics.
* **Specialized Optical Models:** Computes normalized discrete kernels for Gaussian, Cauchy, Airy disk, Moffat, Motion blur, and Scattering profiles.
* **Scientific Data Support:** Processes high-dynamic-range data, including 16-bit TIFF and FITS formats, without loss of precision or clipping.
* **Interactive GUI:** Built-in graphical interface for visual parameter tuning, image loading, and real-time preview of restoration metrics.
* **Automated Benchmarking:** Includes a suite to compare the engine's performance against standard methods (Wiener filters, Gaussian unsharp masks), generating Excel reports and comparative metric plots.
* **Adaptive Stopping Criterion:** Prevents over-deconvolution by monitoring the Ringing Score (Laplacian kurtosis) and relative Tenengrad gain per iteration.

## Mathematical Background

The core of the restoration engine relies on solving the inverse problem of image formation:
`g(x,y) = (h * f)(x,y) + n(x,y)`

Where `g` is the observed image, `h` is the Point Spread Function, `f` is the true signal, and `n` represents noise. The system utilizes:
1. **Richardson-Lucy Algorithm:** An iterative maximum-likelihood approach optimized via Fast Fourier Transform (FFT) convolutions.
2. **Golden-Section Search:** Used for 1D scalar optimization of the PSF scale parameter (gamma) by maximizing the Tenengrad sharpness metric while penalizing ringing artifacts.
3. **Multiscale Unsharp Masking:** Applied as an optional post-processing step for adaptive contrast enhancement based on image standard deviation.

## Repository Structure

* `app/` - Core application files:
  * `diploma_engine.py` - The main restoration engine, PSF models, and optimization logic.
  * `diploma_gui.py` - The Tkinter-based graphical user interface.
* `scripts/` - Utility scripts:
  * `benchmark_comparison.py` - Runs comparative tests against standard algorithms and exports results to `.xlsx`.
  * `psf_1d_code.py` - Generates 1D radial profiles of the supported PSF models.
  * `psf_2d_code.py` - Generates 2D heatmaps of the supported PSF models.
* `data/` - Directory for sample scientific images used for testing and benchmarking.
* `docs/` - Project documentation, generated plots, and the full text of the thesis.
* `requirements.txt` - List of required Python libraries.

## Installation

1. Clone the repository to your local machine:
   ```bash
   git clone [https://github.com/gerseym0/cmos-adaptive-restoration.git](https://github.com/gerseym0/cmos-adaptive-restoration.git)
   cd cmos-adaptive-restoration
(Optional but recommended) Create and activate a virtual environment:

Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
Install the required dependencies:

Bash
pip install -r requirements.txt
Usage
Graphical User Interface
To launch the main application with the graphical user interface, run:

Bash
python app/diploma_gui.py
From the GUI, you can load images from the data/ directory, select the scientific domain, adjust the number of iterations, and save the restored 16-bit output.

Benchmarking
To run the comparative analysis against standard filters:

Bash
python scripts/benchmark_comparison.py
This script will process all images in the test directory, calculate Tenengrad and Ringing Score metrics, and output comparison_results.xlsx and comparison_chart.png.

PSF Visualization
To generate visual representations of the mathematical models:

Bash
python scripts/psf_1d_code.py
python scripts/psf_2d_code.py
Visuals
(Note: Replace the paths below with the actual links to your images once uploaded)

Graphical Interface:

PSF Models (2D Heatmaps):

Benchmarking Results:

Built With
Python 3.10+ - Core programming language.

NumPy & SciPy - High-performance arrays, FFT convolutions, and mathematical optimization.

OpenCV - Image I/O operations and gradient/Laplacian calculations.

Scikit-Image - Reference algorithms for benchmarking.

Astropy - Handling FITS astronomical data formats.

Matplotlib & OpenPyXL - Data visualization and report generation.

Tkinter - Native Python GUI framework.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Academic Context
Author: Ildar Ibatullin

Institution: Taras Shevchenko National University of Kyiv

Faculty: Faculty of Radiophysics, Electronics and Computer Systems

Department: Computer Engineering

Year: 2026
