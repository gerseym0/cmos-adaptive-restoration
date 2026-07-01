# Adaptive Image Restoration for Scientific CMOS Sensors

This repository contains an automated software system for adaptive filtering and image restoration. It is specifically designed for processing scientific images acquired via CMOS sensors in domains such as deep space astrophotography, planetary imaging, remote sensing, and fluorescence microscopy. 

The engine dynamically identifies the nature of optical blur and applies Richardson-Lucy deconvolution optimized with specialized Point Spread Function (PSF) models.

This project was developed as a Bachelor's Thesis in Computer Engineering at Taras Shevchenko National University of Kyiv.

## Key Features

* **Intelligent PSF Identification:** Automatically selects the best-fitting optical model based on domain parameters, utilizing Gradient Kurtosis and Directional Ratio statistics.
* **Specialized Optical Models:** Supports Gaussian, Cauchy, Airy disk, Moffat, Motion blur, and Scattering profiles.
* **Scientific Data Support:** Processes high-dynamic-range data (16-bit TIFF, FITS) without loss of precision.
* **Interactive GUI:** Built-in graphical interface for visual parameter tuning, image loading, and real-time preview of restoration metrics (Tenengrad, Laplacian Variance, Ringing Score).
* **Benchmarking Suite:** Includes automated tools to compare the engine's performance against standard methods (Wiener filters, Gaussian unsharp masks), generating Excel reports and metric plots.

## Repository Structure

* `app/` - Core application files, including the restoration engine (`diploma_engine.py`) and the graphical user interface (`diploma_gui.py`).
* `scripts/` - Utility scripts for generating 1D/2D PSF visualizations and running automated benchmarks (`benchmark_comparison.py`).
* `data/` - Directory for sample scientific images used for testing and benchmarking.
* `docs/` - Project documentation, screenshots, and the full text of the thesis.
* `requirements.txt` - List of required Python libraries.

## Installation

1. Clone the repository to your local machine:
   ```bash
   git clone [https://github.com/gerseym0/cmos-adaptive-restoration.git](https://github.com/gerseym0/cmos-adaptive-restoration.git)
   cd cmos-adaptive-restoration
Install the required dependencies:

Bash
pip install -r requirements.txt
Usage
Graphical Interface
To launch the main application with the graphical user interface, run:

Bash
python app/diploma_gui.py
Benchmarking
To run the comparative analysis against standard filters and generate an .xlsx report and .png charts, run:

Bash
python scripts/benchmark_comparison.py
PSF Visualization
To generate 1D profiles or 2D heatmaps of the supported PSF models:

Bash
python scripts/psf_1d_code.py
python scripts/psf_2d_code.py
Academic Context
Author: Ildar Ibatullin

Institution: Taras Shevchenko National University of Kyiv

Faculty: Faculty of Radiophysics, Electronics and Computer Systems

Department: Computer Engineering
