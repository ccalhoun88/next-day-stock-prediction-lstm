# eval.py
# CIS 732 - Next Day Stock Price Prediction
# evaluates our Naive Baseline, Linear Regression and the Long-Short Term Memory Model this project is based on.
# K-10 TimeSeries Split Cross-validation across 3 random seeds
# Statistical test: One Sided Paired t-test
# Added Hypotheses: H0: LSTM >= Naive Baseline, H1: LSTM < Naive Baseline

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import ttest_rel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# Variable Configuration
data_path = "./data/sp500_stocks.csv"
target_stocks = ["ORCL", "MRK", "PEP"]
features = ["Adj Close", "Open", "High", "Low"]
window = 10
n_features = 4
n_splits = 10
seeds = [42, 99, 123]
epochs = 100
batch_size = 8
patience = 10
alpha         = 0.05
n_comparisons = 6  # 2 comparisons x 3 stocks — Bonferroni correction
alpha_adj     = alpha / n_comparisons

# Seed Helper Logic
def set_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)

# Load the data
def load_data(path, stocks):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    return {stock: df[df["Symbol"] == stock][["Date"] + features].reset_index(drop=True)
            for stock in stocks}

# Create Sequences
# This is used for the Lookback windows that we used in model creation
# Previously used for the 10, 30, 60 windows. Now just 10
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window][0])
    return np.array(X), np.array(y)

# Inverse Transform
# This is used to fix the LSTM model and keep the predictions in scale with the actual
def inverse_transform_target(scaler, values):
    dummy = np.zeros((len(values), n_features))
    dummy[:, 0] = values
    return scaler.inverse_transform(dummy)[:, 0]

# LSTM Model Builder
# Heart of the project
def build_lstm_model(window, n_features):
    model = Sequential()
    model.add(Input(shape=(window, n_features)))
    model.add(LSTM(32, kernel_initializer='glorot_uniform',
                   recurrent_initializer='orthogonal'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
        loss='mse'
    )
    return model

# add 30, 60 day window logic back
def run_single_holdout(stock_data, seed):
    """Evaluates all three models across windows 10, 30, 60 on a single holdout split."""
    set_seeds(seed)
    lookback_windows = [10, 30, 60]
    train_ratio = 0.70
    val_ratio   = 0.15

    holdout_results = {stock: {w: {} for w in lookback_windows}
                       for stock in target_stocks}

    print("\n" + "="*65)
    print("SINGLE HOLDOUT EVALUATION — Windows 10, 30, 60")
    print("="*65)

    for stock in target_stocks:
        data = stock_data[stock][features].values
        n = len(data)
        train_end = int(n * train_ratio)
        val_end   = train_end + int(n * val_ratio)

        train_raw = data[:train_end]
        test_raw  = data[val_end:]

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_raw)
        test_scaled  = scaler.transform(test_raw)

        print(f"\n{stock}:")

        for w in lookback_windows:
            X_train, y_train = create_sequences(train_scaled, w)
            X_test,  y_test  = create_sequences(test_scaled,  w)

            y_test_real = inverse_transform_target(scaler, y_test)

            # Naive Baseline
            y_pred_nb_real = inverse_transform_target(scaler, X_test[:, -1, 0])
            mae_nb   = mean_absolute_error(y_test_real, y_pred_nb_real)
            rmse_nb  = np.sqrt(mean_squared_error(y_test_real, y_pred_nb_real))
            mape_nb  = np.mean(np.abs((y_test_real - y_pred_nb_real) / y_test_real)) * 100

            # Linear Regression
            lr = LinearRegression()
            lr.fit(X_train.reshape(len(X_train), -1), y_train)
            y_pred_lr_real = inverse_transform_target(
                scaler, lr.predict(X_test.reshape(len(X_test), -1)))
            mae_lr   = mean_absolute_error(y_test_real, y_pred_lr_real)
            rmse_lr  = np.sqrt(mean_squared_error(y_test_real, y_pred_lr_real))
            mape_lr  = np.mean(np.abs((y_test_real - y_pred_lr_real) / y_test_real)) * 100

            # LSTM — window 10 only
            if w == 10:
                set_seeds(seed)
                tf.keras.backend.clear_session()
                val_split  = int(len(X_train) * 0.8)
                lstm_model = build_lstm_model(w, n_features)
                lstm_model.fit(
                    X_train[:val_split], y_train[:val_split],
                    validation_data=(X_train[val_split:], y_train[val_split:]),
                    epochs=epochs, batch_size=batch_size,
                    callbacks=[EarlyStopping(monitor='val_loss',
                               patience=patience, restore_best_weights=True)],
                    verbose=0
                )
                y_pred_lstm_real = inverse_transform_target(
                    scaler, lstm_model.predict(X_test, verbose=0).flatten())
                mae_lstm  = mean_absolute_error(y_test_real, y_pred_lstm_real)
                rmse_lstm = np.sqrt(mean_squared_error(y_test_real, y_pred_lstm_real))
                mape_lstm = np.mean(np.abs((y_test_real - y_pred_lstm_real) /
                                           y_test_real)) * 100
            else:
                mae_lstm = rmse_lstm = mape_lstm = None

            holdout_results[stock][w] = {
                "Naive Baseline":     {"MAE": mae_nb,   "RMSE": rmse_nb,   "MAPE": mape_nb},
                "Linear Regression":  {"MAE": mae_lr,   "RMSE": rmse_lr,   "MAPE": mape_lr},
                "LSTM":               {"MAE": mae_lstm, "RMSE": rmse_lstm, "MAPE": mape_lstm}
            }

            print(f"  Window {w}:")
            print(f"    Naive Baseline    — MAE: ${mae_nb:.2f} | "
                  f"RMSE: ${rmse_nb:.2f} | MAPE: {mape_nb:.2f}%")
            print(f"    Linear Regression — MAE: ${mae_lr:.2f} | "
                  f"RMSE: ${rmse_lr:.2f} | MAPE: {mape_lr:.2f}%")
            if w == 10:
                print(f"    LSTM              — MAE: ${mae_lstm:.2f} | "
                      f"RMSE: ${rmse_lstm:.2f} | MAPE: {mape_lstm:.2f}%")
            else:
                print(f"    LSTM              — Not evaluated for window {w} "
                      f"(structural convergence failure — see notebook)")

    return holdout_results

# K-Fold Evaulation
# Needed for true data testing
def run_kfold(stock_data, seed):
    set_seeds(seed)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = {stock: {"Naive Baseline": [], "Linear Regression": [], "LSTM": []}
               for stock in target_stocks}

    for stock in target_stocks:
        print(f"  {stock}...", end=" ")
        X_raw = stock_data[stock][features].values

        for train_idx, test_idx in tscv.split(X_raw):
            train_raw = X_raw[train_idx]
            test_raw  = X_raw[test_idx]

            fold_scaler = MinMaxScaler()
            train_scaled = fold_scaler.fit_transform(train_raw)
            test_scaled  = fold_scaler.transform(test_raw)

            X_train, y_train = create_sequences(train_scaled, window)
            X_test,  y_test  = create_sequences(test_scaled,  window)

            y_test_real = inverse_transform_target(fold_scaler, y_test)

            # Naive Baseline
            y_pred_nb_real = inverse_transform_target(fold_scaler, X_test[:, -1, 0])
            mape_nb = np.mean(np.abs((y_test_real - y_pred_nb_real) / y_test_real)) * 100
            results[stock]["Naive Baseline"].append(mape_nb)

            # Linear Regression
            lr = LinearRegression()
            lr.fit(X_train.reshape(len(X_train), -1), y_train)
            y_pred_lr_real = inverse_transform_target(
                fold_scaler, lr.predict(X_test.reshape(len(X_test), -1)))
            mape_lr = np.mean(np.abs((y_test_real - y_pred_lr_real) / y_test_real)) * 100
            results[stock]["Linear Regression"].append(mape_lr)

            # LSTM
            set_seeds(seed)
            tf.keras.backend.clear_session()
            val_split  = int(len(X_train) * 0.8)
            lstm_model = build_lstm_model(window, n_features)
            lstm_model.fit(
                X_train[:val_split], y_train[:val_split],
                validation_data=(X_train[val_split:], y_train[val_split:]),
                epochs=epochs, batch_size=batch_size,
                callbacks=[EarlyStopping(monitor='val_loss',
                           patience=patience, restore_best_weights=True)],
                verbose=0
            )
            y_pred_lstm_real = inverse_transform_target(
                fold_scaler, lstm_model.predict(X_test, verbose=0).flatten())
            mape_lstm = np.mean(np.abs((y_test_real - y_pred_lstm_real) / y_test_real)) * 100
            results[stock]["LSTM"].append(mape_lstm)

        print("done")
    return results

# Paired T-Test
def run_ttest(results, label=""):
    print("\n" + "="*65)
    print("PAIRED T-TEST RESULTS (alpha = 0.05, two-sided)")
    print(f"Hypothesis: H0: μLSTM >= μNaive | H1: μLSTM < μNaive")
    print(f"Test: One-sided (alternative='less') | α={alpha} | "
          f"Bonferroni adjusted α={alpha_adj:.4f}")
    print("="*65)

    comparisons = [("Linear Regression", "Naive Baseline"),
                   ("LSTM", "Naive Baseline")]

    for stock in target_stocks:
        print(f"\n{stock}:")
        for model_a, model_b in comparisons:
            t_stat, p_value = ttest_rel(
                results[stock][model_a],
                results[stock][model_b],
                alternative='less'
                
            )
            sig       = "SIGNIFICANT" if p_value < alpha_adj else "NOT SIGNIFICANT"
            direction = "better" if np.mean(results[stock][model_a]) < \
                        np.mean(results[stock][model_b]) else "worse"
            print(f"  {model_a} vs {model_b}:")
            print(f"    t={t_stat:.4f} | p={p_value:.4f} | {sig} | {model_a} performs {direction} than {model_b}")

# ── Combined T-Test Across Seeds ────────────────────────────────
def run_combined_ttest(all_seed_results):
    print("\n" + "="*65)
    print("COMBINED T-TEST — Averaged Across All Seeds")
    print(f"Hypothesis: H0: μLSTM >= μNaive | H1: μLSTM < μNaive")
    print(f"Test: One-sided (alternative='less') | "
          f"Bonferroni adjusted α={alpha_adj:.4f}")
    print("="*65)

    comparisons = [("Linear Regression", "Naive Baseline"),
                   ("LSTM", "Naive Baseline")]

    for stock in target_stocks:
        print(f"\n{stock}:")
        for model_a, model_b in comparisons:
            # Average MAPE scores across seeds fold by fold
            combined_a = np.mean([all_seed_results[s][stock][model_a]
                                   for s in seeds], axis=0)
            combined_b = np.mean([all_seed_results[s][stock][model_b]
                                   for s in seeds], axis=0)

            t_stat, p_value = ttest_rel(combined_a, combined_b,
                                         alternative='less')
            sig = "SIGNIFICANT" if p_value < alpha_adj else "NOT SIGNIFICANT"
            direction = "better" if np.mean(combined_a) < \
                        np.mean(combined_b) else "worse"
            print(f"  {model_a} vs {model_b}:")
            print(f"    t={t_stat:.4f} | p={p_value:.4f} | "
                  f"{sig} (Bonferroni α={alpha_adj:.4f}) | "
                  f"{model_a} performs {direction} than {model_b}")

# ── Visualizations ──────────────────────────────────────────────
def save_charts(all_seed_results):
    os.makedirs("./outputs", exist_ok=True)
    models = ["Naive Baseline", "Linear Regression", "LSTM"]
    colors = ["#392F5A", "#9DD9D2", "#FF8811"]
    fold_labels = [f"F{i+1}" for i in range(n_splits)]

    # Individual stock charts — one per stock
    for stock in target_stocks:
        fig, axes = plt.subplots(1, len(seeds), figsize=(18, 5))
        fig.suptitle(f"{stock} — K-Fold MAPE by Fold Across Seeds",
                     fontsize=14)

        for col, seed in enumerate(seeds):
            ax = axes[col]
            for model, color in zip(models, colors):
                mapes = all_seed_results[seed][stock][model]
                ax.plot(fold_labels, mapes, marker="o", label=model,
                        color=color, linewidth=2)
            ax.set_title(f"Seed {seed}")
            ax.set_xlabel("Fold")
            ax.set_ylabel("MAPE (%)")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = f"./outputs/kfold_mape_{stock}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

    # Combined summary chart — all stocks, mean MAPE across seeds
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Mean MAPE Across Seeds — K=10 Fold | Window 10",
                 fontsize=14)

    for idx, stock in enumerate(target_stocks):
        ax = axes[idx]
        for model, color in zip(models, colors):
            mean_mapes = [np.mean(all_seed_results[s][stock][model])
                          for s in seeds]
            ax.bar(seeds, mean_mapes, color=color, label=model,
                   alpha=0.7, width=15)
        ax.set_title(stock)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Mean MAPE (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = "./outputs/kfold_summary_combined.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

# ── Cross-Seed Summary ──────────────────────────────────────────
def print_cross_seed_summary(all_seed_results):
    print("\n" + "="*65)
    print("CROSS-SEED STABILITY SUMMARY")
    print("="*65)

    for stock in target_stocks:
        print(f"\n{stock}:")
        for model in ["Naive Baseline", "Linear Regression", "LSTM"]:
            seed_means = [np.mean(all_seed_results[s][stock][model])
                          for s in seeds]
            print(f"  {model}:")
            print(f"    Per-seed means : {[f'{m:.2f}%' for m in seed_means]}")
            print(f"    Mean across seeds: {np.mean(seed_means):.2f}% | "
                  f"Std: {np.std(seed_means):.2f}%")

# Main Evaluation
if __name__ == "__main__":

    # Print formal hypotheses
    print("="*65)
    print("STATISTICAL HYPOTHESIS")
    print("="*65)
    print("H0: μLSTM >= μNaive")
    print("    The LSTM model does not achieve lower MAPE than the")
    print("    Naive Baseline using 10-day price-based features.")
    print()
    print("H1: μLSTM < μNaive")
    print("    The LSTM model achieves statistically significantly")
    print("    lower MAPE than the Naive Baseline.")
    print()
    print(f"Test  : One-sided paired Student's t-test")
    print(f"Alpha : {alpha} | Bonferroni adjusted: {alpha_adj:.4f}")
    print(f"Method: k={n_splits} TimeSeriesSplit | Seeds: {seeds}")
    print("="*65)

    print("\nLoading data...")
    stock_data = load_data(data_path, target_stocks)

    # Single holdout evaluation — windows 10, 30, 60
    holdout_results = run_single_holdout(stock_data, seeds[0])

    # K-fold evaluation across all seeds
    all_seed_results = {}

    for seed in seeds:
        print(f"\nRunning k-fold with seed {seed}...")
        seed_results = run_kfold(stock_data, seed)
        all_seed_results[seed] = seed_results

        print(f"\nSeed {seed} Summary:")
        for stock in target_stocks:
            print(f"  {stock}:")
            for model, scores in seed_results[stock].items():
                print(f"    {model}: Mean MAPE = {np.mean(scores):.2f}% | "
                      f"Std = {np.std(scores):.2f}%")

        run_ttest(seed_results, label=f"Seed {seed}")

    # Cross-seed summary
    print_cross_seed_summary(all_seed_results)

    # Combined t-test across all seeds
    run_combined_ttest(all_seed_results)

    # Save charts
    print("\nSaving charts...")
    save_charts(all_seed_results)

    print("\nEvaluation complete.")
