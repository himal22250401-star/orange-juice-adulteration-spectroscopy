# UV–Vis Spectroscopic Detection and Quantification of Synthetic Azo Dyes in Orange Juice

**Repository:** `orange-juice-adulteration-spectroscopy`

A reproducible, leakage-audited chemometric and machine learning pipeline for identifying and quantifying synthetic azo dye adulterants (**Tartrazine**, **Sunset Yellow**, and **Metanil Yellow**) in orange juice using UV–Vis absorbance spectroscopy (301–700 nm).

---

## Overview

This repository contains the end-to-end exploratory spectroscopy, preprocessing screening, group-aware validation, and model benchmarking workflow for detecting synthetic food dyes in complex citrus matrices. Every modeling step is encapsulated inside leak-proof pipelines with cryptographic state tracking to guarantee that reference-based scatter corrections, scalers, and hyperparameter searches never leak across physical preparation groups or into the sealed holdout set.

The full analysis — all cell code and complete outputs (full tables and all 600 DPI figures) — is provided three ways under `Code and Output/`: the executable notebook (`.ipynb`), a rendered `.html`, and a print-ready `.pdf`.

---

## Dataset Summary

* **File:** `Dataset/dataset.xlsx` (`Calibration` and `Adulterated` sheets)
* **Spectral Range:** 301–700 nm at 1 nm spacing (400 wavelength predictors)
* **Baseline Correction:** AirPLS applied prior to modeling
* **Sample Composition:** **1,970 spectra** across **97 physical preparation groups** (26 concentration levels, 0.0–24.0 µg/mL)

| Domain | Class | Spectra (n) | Physical Groups | Spectra / Group | Observed Peak λmax (nm) | Reference λmax (nm) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Pure Standard (Calibration)** | Tartrazine | 180 | 9 | 20 | 432 | 425 |
| **Pure Standard (Calibration)** | Sunset Yellow | 200 | 10 | 20 | 486 | 485 |
| **Pure Standard (Calibration)** | Metanil Yellow | 220 | 11 | 20 | 427 | 435 |
| **Matrix (Orange Juice)** | Tartrazine | 500 | 25 | 20 | 435 | 425 |
| **Matrix (Orange Juice)** | Sunset Yellow | 500 | 25 | 20 | 483 | 485 |
| **Matrix (Orange Juice)** | Metanil Yellow | 320 | 16 | 20 | 434 | 435 |
| **Matrix (Orange Juice)** | Control (Pure OJ) | 50 | 1 | 50 | 330 *(UV matrix)* | — |

---

## Pipeline Architecture & Gates

1. **Phase 1 — Exploratory Spectroscopy & Fitness Audit (Cells 00–09):**
   * Verifies 100% finite spectral values (788,000 data points), zero detector clipping (<1.15 A screen), and monotonic Beer–Lambert response (R² = 0.9595–0.9921, Spearman ρ = 0.9804–0.9957).
   * Quantifies matrix-induced bathochromic/hypsochromic peak shifts (+3 nm Tartrazine, -3 nm Sunset Yellow, +7 nm Metanil Yellow) and instrument repeatability (~3.15% median within-group RSD at peak).
   * Identifies a 4.70× UV background elevation at 301 nm in the juice matrix (0.366 A vs. 0.078 A in pure standards).
2. **Phase 2 — Leakage-Safe Preprocessing & Exclusion Audit (Cells 10–13):**
   * Implements custom `scikit-learn` transformers where reference-based scatter corrections (`MSC`, `EMSC`, `PQN`) learn their reference spectrum strictly inside `fit()` on each training fold.
   * Reproduces two pre-registered exclusions: standalone **SNV** (amplifies additive noise by 6.84×, exceeding the 5.0× limit) and **Edge-Anchored Detrend** (drives Metanil Yellow absorbance negative in 100% of matrix spectra, mean = -0.1739 A).
   * Locks 7 retained representations: `Raw`, `SG1`, `SNV+SG1`, `MSC`, `L2`, `EMSC`, and `PQN`.
3. **Phase 3 & 4 — Group-Aware Split, Seal & Leakage Probe (Cells 14–20):**
   * **80:20 Group-Aware Interior Split (Seed 42):** Holds out whole physical preparation groups strictly interior to each dye's concentration range (Tartrazine: `[1, 9, 14, 17, 21]` µg/mL; Sunset Yellow: `[5, 6, 14, 16, 20]` µg/mL; Metanil Yellow: `[3, 10, 13]` µg/mL).
   * **Development / Test Partition:** 53 development groups (1,060 spectra) and 13 sealed test groups (260 spectra).
   * **Repeated Grouped Cross-Validation:** 5 folds × 5 repeats (25 evaluations per combination) using `StratifiedGroupKFold` for classification and concentration-balanced windowed group folds for regression.
   * **Permuted-Label Probe:** Shuffling labels at the group level collapses performance to chance (Macro F1 = 0.309 ± 0.050 for LogReg; R² = −9.080 ± 3.912 for Ridge), proving zero data leakage.
4. **Phase 5 & 6 — Adulterant Identity Classification (Cells 21–31):**
   * Benchmarks **63 combinations** (7 preprocessings × 9 models: `PLS-DA`, `LogReg`, `SVM`, `kNN`, `DecisionTree`, `RandomForest`, `GradientBoosting`, `XGBoost`, `LightGBM`), ranked by independent group-majority vote Macro F1.
   * Refines leading candidates via bounded grid search, Optuna TPE (30 trials), and nested grouped CV probability calibration (`uncalibrated` vs. `sigmoid` vs. `isotonic`).
   * Freezes **`EMSC` + `StandardScaler` + `LogisticRegression` (C = 0.9847, uncalibrated)** and evaluates once on the sealed test set.
5. **Phase 7 — Dye Concentration Regression (Cells 32–39):**
   * Selects PLSR latent variables via the one-standard-error parsimony rule and benchmarks **63 combinations per dye** against single-wavelength univariate OLS calibration at λmax.
   * Demonstrates that reference-based scatter corrections (`MSC`, `EMSC`, `PQN`) and `L2` normalization strip concentration-dependent Beer–Lambert amplitude, whereas **`Raw` (AirPLS) + `PLSR`** consistently leads across all three dyes.
6. **Phase 8–11 — Interpretation, Robustness & Reproducibility Audit (Cells 40–48):**
   * Attribution (PLS-VIP / logistic coefficients), descriptive severity bands over pooled out-of-sample predictions, matrix-effect and domain-transfer probes, single-wavelength LOD/LOQ (slope method), seed/latent-variable robustness, and the final leakage + reproducibility gate (**G9: 12/12 checks PASS**).
   * Assembles the publication figure and table sets with a claim-evidence ledger and a hostile Q1-style review.

---

## Key Results

### 1. Sealed-Test Classification (`EMSC` + `LogisticRegression`)

Evaluated a single time on the 260 held-out test spectra (13 independent preparation groups):

| Class | Precision | Recall | F1-Score | One-vs-Rest ROC-AUC | Test Spectra (n) | Test Groups |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tartrazine** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100 | 5 |
| **Sunset Yellow** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100 | 5 |
| **Metanil Yellow** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 60 | 3 |
| **Overall / Macro** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **260** | **13** |

* **Group-Vote Macro F1:** `1.0000` (13/13 groups)
* **Matthews Correlation Coefficient (MCC):** `1.0000`
* **Multiclass Brier Score:** `7.33e-07` (`0.0000`)
* **Expected Calibration Error (10-bin ECE):** `0.0003`

### 2. Development Cross-Validation Regression (`Raw` + `PLSR`)

Evaluated across 25 concentration-stratified grouped folds on the development set:

| Dye Endpoint | Frozen Pipeline | Selected LVs (One-SE) | RMSECV (µg/mL) | MAE (µg/mL) | Pooled R² | Univariate λmax Benchmark RMSECV (µg/mL) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Tartrazine** | `Raw` + `PLSR` | 3 LV | **0.3323 ± 0.1321** | 0.2593 | 0.9976 | 0.6240 ± 0.0797 (at 434 nm) |
| **Sunset Yellow** | `Raw` + `PLSR` | 2 LV | **0.4628 ± 0.1343** | 0.3764 | 0.9959 | 0.5913 ± 0.0562 (at 483 nm) |
| **Metanil Yellow** | `Raw` + `PLSR` | 3 LV | **0.2042 ± 0.0401** | 0.1547 | 0.9980 | 0.9044 ± 0.1056 (at 434 nm) |

### 3. Sealed-Test Regression (`Raw` + `PLSR`, one evaluation per dye)

| Dye Endpoint | RMSEP (µg/mL) | R² | Slope | Bias (µg/mL) |
| :--- | :---: | :---: | :---: | :---: |
| **Tartrazine** | 0.630 | 0.9917 | 1.03 | +0.29 |
| **Sunset Yellow** | 0.509 | 0.9923 | 0.96 | −0.34 |
| **Metanil Yellow** | 0.307 | 0.9946 | 0.98 | −0.24 |

> **Scope.** The sealed near-perfect identification reflects a genuine, leakage-audited ceiling for this three-dye task; it is reported as a rapid **screening** method, not a validated reference-grade determination (no accredited reference-method comparison was performed). Severity bands are descriptive strata, not legal limits.

---

## Regulatory Context & Severity Bands (Bangladesh)

The intended jurisdiction is **Bangladesh**, where food colourants are governed by the **Bangladesh Food Safety Authority (BFSA)** food-additive regulations (Codex-aligned) and the applicable **Bangladesh Standards and Testing Institution (BSTI)** product standard.

| Dye | INS / identity | Regulatory status for food use |
| :--- | :--- | :--- |
| **Tartrazine** | INS 102 (CI Food Yellow 4) | **Permitted** synthetic colour, subject to a maximum permissible level that depends on the product category. |
| **Sunset Yellow FCF** | INS 110 (CI Food Yellow 3) | **Permitted** synthetic colour, subject to a maximum permissible level that depends on the product category. |
| **Metanil Yellow** | Acid Yellow 36 (non-food azo dye) | **Non-permitted industrial dye** — not an approved food colourant; any detectable amount in food is adulteration, so no permissible level is assigned. |

**Regulatory classification.** Metanil Yellow is a non-permitted industrial dye with no approved food use; it carries no permissible level, and its presence at any detectable concentration constitutes adulteration. Tartrazine and Sunset Yellow FCF are permitted synthetic colourants whose maximum permissible levels are specified per food category under the Bangladesh Food Safety (Food Additives) Regulations and the corresponding BSTI product standard, in line with the Codex General Standard for Food Additives. These maxima differ across the applicable Codex categories — fruit and vegetable juices (14.1.2), nectars (14.1.3), and water-based flavoured drinks (14.1.4) — and added colourants are generally not permitted in standardized 100 % fruit juice. The governing compliance threshold is therefore defined by the specific product category, and predicted concentrations are evaluated against the category-appropriate limit rather than a single universal value.

**Severity-band stratification.** Predicted concentrations are stratified into descriptive severity bands spanning the experimentally covered range (dilution factor 12.5×; product ppm = analytical µg mL⁻¹ × 12.5). The bands constitute a post-prediction reporting layer, independent of model training, and provide an interpretive scale of adulteration magnitude rather than a regulatory classification. Predictive behaviour is characterised at the band boundaries, particularly the 5–8 µg mL⁻¹ transition region, where adjacent-band confusion is most likely to arise.

Tartrazine & Sunset Yellow:

| Band | Analytical (µg mL⁻¹) | Original-product equivalent (ppm) |
| :--- | :---: | :---: |
| A | 0–4 | 0–50 |
| B | 5–8 | 62.5–100 |
| C | 9–13 | 112.5–162.5 |
| D | 14–18 | 175–225 |
| E | 19–24 | 237.5–300 |

Metanil Yellow:

| Band | Analytical (µg mL⁻¹) | Original-product equivalent (ppm) |
| :--- | :---: | :---: |
| M1 | 0–3 | 0–37.5 |
| M2 | 4–7 | 50–87.5 |
| M3 | 8–11 | 100–137.5 |
| M4 | 12–15 | 150–187.5 |

Over pooled out-of-sample predictions (development out-of-fold combined with the sealed test), band assignment reproduces with **0.96 accuracy (A–E)** and **0.99 accuracy (M1–M4)**, the few errors being confined to adjacent-band boundaries such as the 5–8 µg mL⁻¹ region. Because predictions are expressed as absolute concentrations (µg mL⁻¹), they map directly onto the category-appropriate BFSA/BSTI maximum level, enabling a compliance assessment for each sample.

**Analytical validation and limitations.** Limits of detection and quantification were determined at the analytical wavelength by the slope method and are reported separately from the multivariate calibrations, whose lower-range performance is characterised through prediction error, bias, precision, and residual structure in the lowest concentration band. Replicate precision, obtained from 20 within-group instrument frames, reflects instrument repeatability; between-preparation reproducibility was not evaluated, as a single physical preparation was produced per concentration level. Metanil Yellow produced approximately one third of the peak response of the other two colourants (0.172 A versus 0.468 A and 0.477 A in the matrix domain), accounting for its comparatively lower analytical sensitivity. All three colourants were analytical-grade standards (Sigma-Aldrich; certified purity ≥ 95 %, weighed mass corrected for purity): Tartrazine (INS 102, CAS 1934-21-0), Sunset Yellow FCF (INS 110, CAS 2783-94-0), and Metanil Yellow (Acid Yellow 36, CAS 587-98-4).

---

## Repository Structure

```text
├── Dataset/
│   └── dataset.xlsx                  # Validated UV-Vis spectral dataset (Calibration + Adulterated)
├── Code and Output/
│   ├── Orange_Juice.ipynb            # Main end-to-end Jupyter notebook (49 cells)
│   ├── Orange_Juice.html             # Rendered notebook with full code + outputs
│   └── Orange_Juice.pdf              # Print-ready notebook (full tables + all figures)
├── Figure/                           # 600 DPI publication figures (PNG), sequential 1–30
├── Tables/                           # Archived analysis tables (XLSX)
├── Checkpoints/                      # Hash-validated cell checkpoints (.pkl) + project_state.json
├── Models/                           # Frozen pipelines (.joblib) + JSON freeze records
├── oj_analysis/                      # Project state & sealed-test audit guard (required to re-run)
│   ├── __init__.py
│   ├── audit.py
│   ├── config.py
│   └── state.py
├── requirements.txt                  # Python package dependencies
└── README.md
```

> `oj_analysis/` and `requirements.txt` are required to re-execute the notebook; the serialized classifier in `Models/` additionally relies on the custom `EMSC` transformer defined in the notebook.

---

## Getting Started & Reproduction

### 1. Clone the Repository

```bash
git clone https://github.com/himal22250401-star/orange-juice-adulteration-spectroscopy.git
cd orange-juice-adulteration-spectroscopy
```

### 2. Install Dependencies

**Python 3.12** is recommended (the gradient-boosting and Optuna wheels used here are published for 3.12). Install the core scientific and gradient-boosting stack:

```bash
pip install numpy pandas scipy scikit-learn matplotlib openpyxl joblib xgboost lightgbm optuna ipython jupyterlab
```

> XGBoost, LightGBM, and Optuna are required for the full 9-model / 63-combination benchmark. Without them the classification benchmark degrades to 7 models / 49 combinations.

### 3. Run the Analysis

1. The notebook expects the project folders (`Dataset/`, `Figures/`, `Tables/`, `Checkpoints/`, `Models/`, `oj_analysis/`) relative to its working directory. Because the notebook ships inside `Code and Output/`, either copy it to the repository root before running, or set `PROJECT_ROOT` in **Cell 00** to the repository root so it can locate those folders.
2. Launch JupyterLab (`jupyter lab`) and run `Orange_Juice.ipynb` via **Restart Kernel and Run All Cells**.
3. Cached benchmark checkpoints in `Checkpoints/` allow rapid re-rendering of tables and 600 DPI figures; delete the `.pkl` files in `Checkpoints/` to force full recomputation of the 63-combination cross-validation grids from scratch.

> Prefer to just read the results? Open `Code and Output/Orange_Juice.html` in a browser, or `Code and Output/Orange_Juice.pdf`, for the complete code and outputs without running anything.

---

## License

This project is licensed under the MIT License — see the [LICENSE](https://github.com/himal22250401-star/orange-juice-adulteration-spectroscopy/blob/main/LICENSE) file for details.
