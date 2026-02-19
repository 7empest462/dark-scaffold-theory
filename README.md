# Pre-Existing Dark Scaffold Cosmology

**A Computational Framework for Pre-Inflationary Dark Matter Structure**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18464392.svg)](https://doi.org/10.5281/zenodo.18464392)

## 📄 Abstract

The **Pre-Existing Dark Scaffold (PEDS)** theory proposes that dark matter existed as a structured cosmic web prior to the Big Bang. This framework interprets the Big Bang as a baryonic injection into a pre-existing gravitational potential, allowing for immediate structure formation without the hierarchical assembly delay required by $\Lambda$CDM.

While the model demonstrates superior performance in resolving high-redshift structural anomalies (JWST, SMBH seeds), it currently exhibits significant tension with the CMB power spectrum ($\Delta \text{BIC} = 5420.1$). This repository provides the N-body simulation suite used to validate the theory's structural predictions.

## 🔭 Physical Verification Results

The PEDS model has been tested against three major cosmological tensions using high-resolution simulations:

| Anomaly | $\Lambda$CDM Prediction | PEDS Result | Significance |
| :--- | :--- | :--- | :--- |
| **Early Galaxy Count (z=15)** | ~0.5 galaxies | **683 galaxies** | Explains JWST "Impossible" Galaxies |
| **Void Baryon Density** | < 10% | **85%** | Unique "Seepage" Signature |
| **SMBH Infall Rate** | < 0.05 $M_\odot$/yr | **0.19 $M_\odot$/yr** | Triggers Direct Collapse (DCBH) |

## 💻 Simulation Suite

This repository contains the Python implementation of the PEDS framework:

### 1. Structural Analysis (`jwst_test.py`, `void_test.py`)
- **JWST Test**: Validates the overabundance of massive galaxies ($>10^{10} M_\odot$) at $z > 10$.
- **Void Test**: Detects the "Dirty Void" signature—a high baryon fraction in cosmic voids resulting from the seepage mechanism.

### 2. Analytical Core (`theory_likelihood.py`, `theory_optimizer.py`)
- **Likelihood Engine**: Performs $\chi^2$ and BIC analysis against observational priors.
- **Optimization**: Uses gradient descent to find preferred values for the scaffold coupling constant ($g_{\phi m}$).

### 3. Physical Diagnostics (`smbh_test.py`, `phase_space_test.py`)
- **SMBH Test**: Models gas infall into scaffold nodes to check for Direct Collapse conditions.
- **Phase Space**: Analyzes the velocity dispersion of early-forming halos.

## 🚀 Usage

```bash
# Install dependencies
pip install numpy matplotlib scipy

# Run the high-resolution simulation pipeline
./run_hires.sh
```

## 📜 Documentation

- [Theory Document](theory_document.md): Detailed physical derivation and hypothesis.
- [LaTeX Manuscript](paper.tex): Academic paper draft with full statistical results.

---
*Research and Simulation Suite by Rob Simens*
