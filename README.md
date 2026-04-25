# Next-Day Stock Price Prediction Using LSTM

**Author:** Diante Calhoun  
**Course:** CIS 732 — Machine Learning and Probabilistic Models  
Kansas State University  
**Date:** Spring 2026

---

## Project Overview

This project evaluates how accurately Machine Learning models can predict 
next-day Adjusted Close stock prices using only price-based 
features across three S&P 500 sectors. The initial aim of the project was to 
see how accurate the LSTM model could perform based on windows of 10, 30 and 60 days.
This was adjusted based on the outcomes.

Three models are compared:

- **Naive Baseline** — persistence model (today's price = tomorrow's prediction)
- **Linear Regression** — Basic flattened sequence input via scikit-learn
- **LSTM** — single-layer Long Short-Term Memory network via TensorFlow/Keras

Models are evaluated using k=10 TimeSeriesSplit cross-validation across 
three random seeds to ensure statistical robustness. A paired Student's 
t-test is used to determine statistical significance.

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
| LSTM | ~3.00% | ~2.87% | ~3.04% |

The Naive Baseline outperformed both ML models across all three sectors,  
consistent with the Efficient Market Hypothesis weak form. All differences  
were statistically significant (p < 0.05) except Linear Regression vs  
Naive Baseline on MRK (p = 0.0685).

---

## Statistical Testing

- **Method:** Two-sided paired Student's t-test (`scipy.stats.ttest_rel`)
- **Resampling:** k=10 TimeSeriesSplit cross-validation
- **Seeds tested:** 42, 99, 123
- **Alpha:** 0.05 with Bonferroni correction (adjusted α = 0.0083)

---

## GenAI Disclosure

Claude.ai (Anthropic),  ChatGPT (OpenAI), and Gemini (Google) were used as development 
assistants throughout this project for debugging, code review, and 
conceptual explanation. All architectural decisions, analysis, and written 
content were authored and reviewed by the project author. A full GenAI 
audit is included in the project report appendix.

