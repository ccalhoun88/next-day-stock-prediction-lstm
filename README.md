# Next-Day Stock Price Prediction Using LSTM

**Author:** Diante Calhoun  
**Course:** CIS 732 — Machine Learning and Probabilistic Models  
Kansas State University  
**Date:** Spring 2026

---

## Project Overview

This project evaluates how accurately Machine Learning models can predict 
next-day Adjusted Close stock prices using only historical price-based 
features across three S&P 500 sectors. 

Three models are compared:

- **Naive Baseline** — persistence model (today's price = tomorrow's prediction)
- **Linear Regression** — Basic flattened sequence input via scikit-learn
- **LSTM** — single-layer Long Short-Term Memory network via TensorFlow/Keras

Models are evaluated using k=10 TimeSeriesSplit cross-validation across 
three random seeds (42, 99, 123) to ensure statistical robustness and 
reproducibility. A one-sided paired Student's t-test with Bonferroni 
correction is used to formally evaluate the statistical hypothesis.

---

## Statistical Hypothesis

**H0:** μLSTM ≥ μNaive — The LSTM model does not achieve lower MAPE than the Naive Baseline  
**H1:** μLSTM < μNaive — The LSTM model achieves statistically significantly lower MAPE  
**Test:** One-sided paired Student's t-test (`scipy.stats.ttest_rel`, `alternative='less'`)  
**Alpha:** 0.05 with Bonferroni correction (adjusted α = 0.0083 across 6 comparisons)

---

## Stocks Evaluated

| Ticker | Company | Sector |
|--------|---------|--------|
| ORCL | Oracle Corporation | Technology |
| MRK | Merck & Co. | Healthcare |
| PEP | PepsiCo Inc. | Consumer Staples |
All tickers have at least 10 years of performance information
---

## Research Question

How accurately can ML models predict next-day Adjusted Close price using 
only the previous 10, 30 and 60 days of price-based features (Adjusted Close, Open, 
High, Low), and does performance vary across sectors?

---

## Repository Structure
next-day-stock-prediction-lstm/
├── ccalhounMLProject.ipynb   # Full development notebook
├── eval.py                   # Reproducible evaluation script
├── requirements.txt          # Python dependencies
├── outputs/                  # Generated charts and visualizations
│     ├── kfold_mape_ORCL.png
│     ├── kfold_mape_MRK.png
│     ├── kfold_mape_PEP.png
│     └── kfold_summary_combined.png
└── README.md

## Dataset

This project uses the S&P 500 Stocks dataset from Kaggle:

**Source:** https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks

**Download instructions:**
1. Visit the link above and log in to Kaggle
2. Click Download to get the zip file
3. Unzip and place `sp500_stocks.csv` in a `/data` folder in the project root

The dataset covers daily OHLCV data from 2010–2024.

## Environment Setup

**Python version:** 3.11

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## How to run

### Option 1 - Evaluation Script
To reproduce the k-fold evaluation and p-values reported in the paper:
```bash
python eval.py
```

**The script will:**
1. Print formal statistical hypotheses (H0 and H1)
2. Run single holdout evaluation across Windows 10, 30, and 60
3. Run k=10 TimeSeriesSplit cross-validation across seeds 42, 99, and 123
4. Print per-seed summaries and paired t-test results with Bonferroni correction
5. Print cross-seed stability summary
6. Run combined t-test averaged across all three seeds
7. Save per-stock fold-by-fold MAPE charts and combined summary chart to ./outputs/

**Note:** LSTM is evaluated on Window 10 only. Windows 30 and 60 experienced 
structural convergence failure during development — see notebook for details.

**Expected output:**
- Per-seed k-fold MAPE summary for all three stocks and models
- Paired t-test results with p-values per stock
- Cross-seed stability summary

### Option 2 - Jupyter Notebook
Open `ccalhounMLProject.ipynb` in Jupyter and run cells sequentially.  
The notebook documents the full development process including exploratory 
analysis, hyperparameter tuning, and iterative model improvements.

---

## Key Results

| Model | ORCL MAPE | MRK MAPE | PEP MAPE |
|-------|-----------|----------|----------|
| Naive Baseline | ~1.12% | ~0.92% | ~0.74% |
| Linear Regression | ~1.16% | ~0.98% | ~0.77% |
| LSTM (avg across seeds) | 3.21% | 3.25% | 3.19% |

The Naive Baseline outperformed both ML models across all three sectors. 
The null hypothesis could not be rejected under the one-sided paired t-test 
framework — p-values ranged from 0.9657 to 0.9999, confirming that neither 
the LSTM nor Linear Regression achieved statistically significantly lower 
MAPE than the persistence baseline. Results are consistent with the 
Efficient Market Hypothesis weak form.

---

## Statistical Testing

- **Method:** One-sided paired Student's t-test (`scipy.stats.ttest_rel`)
- **Direction:** `alternative='less'` (testing if model MAPE < Naive MAPE)
- **Resampling:** k=10 TimeSeriesSplit cross-validation
- **Seeds tested:** 42, 99, 123
- **Alpha:** 0.05 with Bonferroni correction (adjusted α = 0.0083)

---

## GenAI Disclosure

Claude.ai (Anthropic) and ChatGPT (OpenAI) were used as development 
assistants throughout this project for debugging, code review, conceptual 
explanation, and report structure guidance. All architectural decisions, 
analysis, and written content were authored and reviewed by the project 
author. A full GenAI audit is included in the project report appendix.

## Contact

Diante Calhoun  
Kansas State University  
CIS 732 — Spring 2026
