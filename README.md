# Comparative insights: fuzzy clustering versus archetypal analysis in vector quantization for Blosc2

This repository contains the code and experiments for the research article:

> **Comparative insights: fuzzy clustering versus archetypal analysis in vector quantization for Blosc2**
>
> Aleix Alcacer and Irene Epifanio
>
> *Universitat Jaume I, Spain / ValgrAI*

## Abstract

Blosc2 is a high-performance compression library and data format designed for binary data such as numerical arrays, tensors, and other structured types. In this work, we develop two new codecs for Blosc2 by leveraging its extensible plugin-based codec framework. Specifically, we integrate **Archetypal Analysis (AA)** and **Fuzzy Clustering (CMeans)** as novel codecs within Blosc2.

Our experiments on the Olivetti Faces dataset demonstrate that AA outperforms Fuzzy Clustering in preserving fine-grained data details, achieving substantially higher Structural Similarity Index (SSIM). These results underscore AA's capability to capture the complete data distribution, including its extreme values, which is essential for achieving high-fidelity compression.

## Repository Structure

```
ra-blosc-codecs/
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── models/
│   ├── __init__.py     # Model wrappers (AA, KMeans, CMeans)
│   └── cmeans.py       # Fuzzy C-Means implementation
└── examples/
    └── models.ipynb    # Main experiment notebook
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aleixalcacer/ra-blosc-codecs.git
cd ra-blosc-codecs
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Experiments

The main experiments are contained in the Jupyter notebook `examples/models.ipynb`. To run them:

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `examples/models.ipynb`

3. Run all cells to reproduce the results

The notebook will:
- Train CMeans and AA models on the Olivetti Faces dataset
- Register custom Blosc2 codecs using the trained models
- Compress and decompress the images
- Compute SSIM metrics and compression ratios
- Generate comparison visualizations

## Results

| Method  | Min  | P25  | P75  | Max  | Mean | Std  |
|---------|------|------|------|------|------|------|
| CMeans  | 0.49 | 0.68 | 0.75 | 0.82 | 0.71 | 0.06 |
| AA      | 0.80 | 0.90 | 0.94 | 0.97 | 0.92 | 0.03 |

*Table: Summary statistics of SSIM values for each reconstruction method*

## Acknowledgments

This work was partially supported by the Spanish Ministry of Science and Innovation PID2022-141699NB-I00 and PID2020-118763GA-I00 and Generalitat Valenciana CIPROM/2023/66.

## License

This is an open access article under the terms of the Creative Commons Attribution License.
