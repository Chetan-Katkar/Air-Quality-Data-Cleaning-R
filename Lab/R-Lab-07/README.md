# R Lab 07 — Customer Segmentation and Predictive Analytics Using Machine Learning

**R Programming Laboratory · Problem Statement 07**
**COs:** CO3, CO4, CO5, CO6 · **LOs:** LO 4.2, 5.1, 5.2, 6.1, 6.2, 6.3

---

## Problem Statement

> An e-commerce company wants to better understand its customers and improve its marketing
> decisions. Using historical online retail transaction data, develop a machine-learning-based
> analytics solution that identifies meaningful customer segments and predicts whether a customer
> is likely to belong to a high-value customer category.

**Dataset:** [UCI Machine Learning Repository — Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
(541,909 transactions, 4,372 customers, 4,070 products, 38 countries, Dec 2010 – Dec 2011)

---

## Contents

| File | Description |
|---|---|
| `R_Lab_07_Customer_Segmentation.ipynb` | **Main deliverable** — executed R notebook with all outputs embedded (58 code cells, 23 plots) |
| `R-Lab-07.R` | Standalone R script version of the same analysis (`Rscript R-Lab-07.R`) |
| `R_LAB_REPORT_07.pdf` | 24-page lab report: methodology, results, figures, interpretation, conclusion |
| `outputs/` | 23 generated figures (PNG) + `customer_segments.csv`, `cluster_profiles.csv`, `model_results.csv`, `results.json` |
| `screenshots/` | 12 console-output captures from the R run |
| `logs/console_output.log` | Full console transcript of the script run |

---

## How to run

### Google Colab (recommended)

1. Upload `R_Lab_07_Customer_Segmentation.ipynb` to [Colab](https://colab.research.google.com).
2. **Runtime → Change runtime type → R**.
3. **Runtime → Run all.** (~3–5 minutes, most of it package installation.)

Colab has full internet access, so the notebook downloads the genuine UCI dataset automatically.

### Locally

```bash
Rscript R-Lab-07.R        # writes everything into ./outputs/
```

Requires: `ggplot2 dplyr tidyr cluster factoextra randomForest e1071 caret pROC plotly readxl corrplot dendextend gridExtra scales`

---

## Pipeline

```
Raw transactions (541,909 rows)
  → EDA: missing values, cancellations, price/quantity anomalies
  → Preprocessing: drop unattributable rows, cancellations, non-positive price/qty, duplicates
  → Feature engineering: RFM + 6 derived behaviours, one row per customer
  → Winsorise (1st/99th pct) → log1p → z-score
  → K-Means (Elbow + Silhouette)  vs  Hierarchical (Ward.D2 + dendrogram)
  → PCA → 2D projection, biplot, interactive 3D Plotly
  → Cluster profiling → 4 named segments
  → Random Forest + SVM → high-value prediction
  → Leakage audit → forward-looking re-evaluation
  → Segment-wise marketing recommendations
```

---

## Deliverables checklist

| # | Deliverable | Location |
|---|---|---|
| 1 | Preprocessing and customer-level feature engineering | Notebook §3–§4 |
| 2 | Elbow Method graph | `outputs/07_elbow_method.png` |
| 3 | K-Means customer segmentation | Notebook §6 |
| 4 | Hierarchical clustering and dendrogram | `outputs/10_dendrogram.png` |
| 5 | Silhouette Score and clustering comparison | `outputs/08`, `outputs/09`, `outputs/11` |
| 6 | PCA-based 2D visualisation | `outputs/13_pca_2d_segments.png` |
| 7 | Cluster-wise profiling and interpretation | Notebook §9, `outputs/15`, `outputs/16` |
| 8 | Random Forest and SVM models | Notebook §10 |
| 9 | Accuracy / Precision / Recall / F1 / ROC-AUC / Confusion Matrix | Notebook §11, `outputs/17`–`19` |
| 10 | Feature importance analysis | Notebook §12, `outputs/20` |
| 11 | Interactive 3D Plotly visualisation | Notebook §8.3 (static: `outputs/23`) |
| 12 | Segment-wise marketing recommendations | Notebook §13 |

---

## Key results

**Segments (K = 4)**

| Segment | Customers | % of base | % of revenue | Median R / F / M |
|---|---|---|---|---|
| Champions / High-Value | 708 | 16.3% | **77.3%** | 14 d / 14 orders / 8,203 |
| Loyal Customers | 1,522 | 35.1% | 18.2% | 28 d / 6 orders / 1,531 |
| Potential / Occasional | 1,483 | 34.2% | 4.3% | 81 d / 3 orders / 370 |
| At-Risk / Dormant | 627 | 14.4% | 0.3% | 99 d / 1 order / 60 |

A clear Pareto pattern: **16.3% of customers generate 77.3% of revenue.**

**Clustering quality** — K-Means silhouette 0.2791 vs Hierarchical (Ward.D2) 0.2156, with 69.2%
agreement between the two methods. **PCA** — PC1 + PC2 explain 87.0% of variance; PC1 is a single
"customer value" axis, PC2 a "lifecycle stage" axis.

**Prediction of high-value membership**

| Setting | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Same-period | Random Forest | 0.9939 | 0.9766 | 0.9858 | 0.9812 | 0.9996 |
| Same-period | SVM (RBF) | 0.9969 | 0.9906 | 0.9906 | 0.9906 | 0.9999 |
| **Forward-looking** | **Random Forest** | **0.9597** | **0.9326** | **0.8491** | **0.8889** | **0.9807** |
| **Forward-looking** | **SVM (RBF)** | **0.9624** | **0.9521** | **0.8443** | **0.8950** | **0.9680** |

> ### Why there are two sets of numbers
>
> The `HighValue` label is *derived* from K-Means run on the same behavioural features, so the
> same-period models are largely re-learning a deterministic function of their own inputs — **target
> leakage by construction**. Notebook §12B re-runs the comparison in a fair setting: features from
> only the first 60% of the year, label from year-end status. F1 falls by ~9% and Recall by ~14
> points. **The forward-looking figures are the honest, deployable ones.**

**Feature importance** — `TotalQuantity` and `UniqueProducts` rank far above `Recency` and
`Tenure`, so depth of catalogue engagement predicts value better than recency does. An
equally-weighted RFM score is a suboptimal value proxy for this business.

---

## Note on the data source

The environment used to generate the committed figures had **no outbound network access to
`archive.ics.uci.edu`**, so the notebook's documented fallback simulator was used. It reproduces
the real dataset's schema, scale (541,909 rows / 4,372 customers / 4,070 products) and
data-quality defects — ~25% missing `CustomerID`, ~1.7% cancellations, negative quantities,
zero/negative prices, missing descriptions — so every preprocessing, clustering and modelling step
is exercised exactly as it would be on the real file.

**Running the notebook in Colab downloads the genuine UCI dataset automatically.** The Section 1
cell output always records which source was used for that run.
