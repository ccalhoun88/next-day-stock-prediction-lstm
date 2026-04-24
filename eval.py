# eval.py
# CIS 732 - Next Day Stock Price Prediction
# evaluates our Naive Baseline, Linear Regression and the Long-Short Term Memory Model this project is based on.
# K-10 TimeSeries Split Cross-validation across 3 random seeds

import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
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
    return {stock: df[df["Symbol"] == stock][["Date"] + FEATURES].reset_index(drop=True)
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
def inverse_transform_target(scaler, values, n_features):
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

# K-Fold Evaulation
# Needed for true data testing
def run_kfold(stock_data, seed):
    set_seeds(seed)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    results = {stock: {"Naive Baseline": [], "Linear Regression": [], "LSTM": []}
               for stock in TARGET_STOCKS}

    for stock in TARGET_STOCKS:
        print(f"  {stock}...", end=" ")
        X_raw = stock_data[stock][FEATURES].values

        for train_idx, test_idx in tscv.split(X_raw):
            train_raw = X_raw[train_idx]
            test_raw  = X_raw[test_idx]

            fold_scaler = MinMaxScaler()
            train_scaled = fold_scaler.fit_transform(train_raw)
            test_scaled  = fold_scaler.transform(test_raw)

            X_train, y_train = create_sequences(train_scaled, WINDOW)
            X_test,  y_test  = create_sequences(test_scaled,  WINDOW)

            y_test_real = inverse_transform_target(fold_scaler, y_test, N_FEATURES)

            # Naive Baseline
            y_pred_nb_real = inverse_transform_target(fold_scaler, X_test[:, -1, 0], N_FEATURES)
            mape_nb = np.mean(np.abs((y_test_real - y_pred_nb_real) / y_test_real)) * 100
            results[stock]["Naive Baseline"].append(mape_nb)

            # Linear Regression
            lr = LinearRegression()
            lr.fit(X_train.reshape(len(X_train), -1), y_train)
            y_pred_lr_real = inverse_transform_target(
                fold_scaler, lr.predict(X_test.reshape(len(X_test), -1)), N_FEATURES)
            mape_lr = np.mean(np.abs((y_test_real - y_pred_lr_real) / y_test_real)) * 100
            results[stock]["Linear Regression"].append(mape_lr)

            # LSTM
            set_seeds(seed)
            tf.keras.backend.clear_session()
            val_split  = int(len(X_train) * 0.8)
            lstm_model = build_lstm_model(WINDOW, N_FEATURES)
            lstm_model.fit(
                X_train[:val_split], y_train[:val_split],
                validation_data=(X_train[val_split:], y_train[val_split:]),
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                callbacks=[EarlyStopping(monitor='val_loss',
                           patience=PATIENCE, restore_best_weights=True)],
                verbose=0
            )
            y_pred_lstm_real = inverse_transform_target(
                fold_scaler, lstm_model.predict(X_test, verbose=0).flatten(), N_FEATURES)
            mape_lstm = np.mean(np.abs((y_test_real - y_pred_lstm_real) / y_test_real)) * 100
            results[stock]["LSTM"].append(mape_lstm)

        print("done")
    return results

# Paired T-Test
def run_ttest(results):
    print("\n" + "="*65)
    print("PAIRED T-TEST RESULTS (alpha = 0.05, two-sided)")
    print("="*65)

    comparisons = [("Linear Regression", "Naive Baseline"),
                   ("LSTM", "Naive Baseline")]

    for stock in TARGET_STOCKS:
        print(f"\n{stock}:")
        for model_a, model_b in comparisons:
            t_stat, p_value = ttest_rel(
                results[stock][model_a],
                results[stock][model_b]
            )
            sig       = "SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
            direction = "worse" if np.mean(results[stock][model_a]) > \
                        np.mean(results[stock][model_b]) else "better"
            print(f"  {model_a} vs {model_b}:")
            print(f"    t={t_stat:.4f} | p={p_value:.4f} | {sig} | {model_a} is {direction}")

# Main Evaluation
if __name__ == "__main__":
    print("Loading data...")
    stock_data = load_data(DATA_PATH, TARGET_STOCKS)

    all_seed_results = {}

    for seed in SEEDS:
        print(f"\nRunning k-fold with seed {seed}...")
        seed_results = run_kfold(stock_data, seed)
        all_seed_results[seed] = seed_results

        print(f"\nSeed {seed} Summary:")
        for stock in TARGET_STOCKS:
            print(f"  {stock}:")
            for model, scores in seed_results[stock].items():
                print(f"    {model}: Mean MAPE = {np.mean(scores):.2f}% | "
                      f"Std = {np.std(scores):.2f}%")

        run_ttest(seed_results)

    print("\n\nCross-Seed Stability:")
    for stock in TARGET_STOCKS:
        print(f"\n{stock}:")
        for model in ["Naive Baseline", "Linear Regression", "LSTM"]:
            seed_means = [np.mean(all_seed_results[s][stock][model]) for s in SEEDS]
            print(f"  {model}: {[f'{m:.2f}%' for m in seed_means]} | "
                  f"Mean={np.mean(seed_means):.2f}% | Std={np.std(seed_means):.2f}%")
