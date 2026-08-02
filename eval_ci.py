import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Safe for headless CI servers
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

def run_sanity_tests(df):
    print("--- RUNNING SANITY TESTS ON FEATURES ---")
    assert df['rolling_avg_10'].isnull().sum() == 0, "Null values in rolling_avg_10"
    assert df['volume_sum_10'].isnull().sum() == 0, "Null values in volume_sum_10"
    assert (df['rolling_avg_10'] > 0).all(), "rolling_avg_10 has non-positive values"
    assert (df['volume_sum_10'] >= 0).all(), "volume_sum_10 has negative values"
    print("ALL FEATURE SANITY TESTS PASSED SUCCESSFULLY!")

def evaluate_and_report():
    csv_files = glob.glob("data/*.csv")
    all_dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df.set_index('timestamp', inplace=True)
        df.ffill(inplace=True)
        
        df['rolling_avg_10'] = df['close'].rolling(window='10min', min_periods=1).mean()
        df['volume_sum_10'] = df['volume'].rolling(window='10min', min_periods=1).sum()
        
        df['close_5min_future'] = df['close'].shift(-5)
        df['target'] = (df['close_5min_future'] > df['close']).astype(int)
        
        df.dropna(subset=['rolling_avg_10', 'volume_sum_10', 'target'], inplace=True)
        df.drop(columns=['close_5min_future'], inplace=True)
        df.reset_index(inplace=True)
        all_dfs.append(df)
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    test_df = full_df.tail(2000).copy()
    
    run_sanity_tests(test_df)
    
    # Connect to the natively generated CI MLflow database
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_uri = "models:/stock_movement_predictor/1"
    print(f"Loading best model directly from MLflow Registry: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    
    X_test = test_df[['rolling_avg_10', 'volume_sum_10']]
    y_test = test_df['target']
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"--- TEST METRICS --- | Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    plt.figure(figsize=(6, 4))
    plt.bar(['Accuracy', 'Precision', 'Recall', 'F1-Score'], [acc, prec, rec, f1], color=['blue', 'green', 'orange', 'red'])
    plt.ylim(0, 1.0)
    plt.title("Test Dataset Evaluation Metrics")
    plt.savefig("metrics_plot.png")
    plt.close()
    
    with open("report.md", "w") as f:
        f.write("# Model CI Evaluation & CML Report\n\n")
        f.write("## 1. Feature Sanity Test Results\n")
        f.write("- **rolling_avg_10:** Non-null & positive value check **PASSED**\n")
        f.write("- **volume_sum_10:** Non-null & non-negative value check **PASSED**\n\n")
        f.write("## 2. Test Dataset Metrics (MLflow Model Registry: Version 1)\n")
        f.write("| Metric | Value |\n| :--- | :--- |\n")
        f.write(f"| Accuracy | `{acc:.4f}` |\n| Precision | `{prec:.4f}` |\n")
        f.write(f"| Recall | `{rec:.4f}` |\n| F1-Score | `{f1:.4f}` |\n\n")
        f.write("## 3. Evaluation Plot\n\n![Metrics Plot](./metrics_plot.png)\n")
        
    print("CML Report generated successfully!")

if __name__ == "__main__":
    evaluate_and_report()
