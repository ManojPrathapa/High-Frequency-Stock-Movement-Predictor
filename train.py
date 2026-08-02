import argparse
import glob
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def prepare_data_from_csvs():
    csv_files = glob.glob("data/*.csv")
    all_dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        ticker = os.path.basename(file).split('__')[0]
        df['stock_name'] = ticker
        df = df.sort_values('timestamp').reset_index(drop=True)
        df.set_index('timestamp', inplace=True)
        df.ffill(inplace=True)
        
        # 10-minute rolling features
        df['rolling_avg_10'] = df['close'].rolling(window='10min', min_periods=1).mean()
        df['volume_sum_10'] = df['volume'].rolling(window='10min', min_periods=1).sum()
        
        # Target: 5 mins into future
        df['close_5min_future'] = df['close'].shift(-5)
        df['target'] = (df['close_5min_future'] > df['close']).astype(int)
        
        df.dropna(subset=['rolling_avg_10', 'volume_sum_10', 'target'], inplace=True)
        df.drop(columns=['close_5min_future'], inplace=True)
        df.reset_index(inplace=True)
        all_dfs.append(df)
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    return full_df

def run_iteration(iteration_num):
    print(f"\n==============================================")
    print(f"   RUNNING TRAINING ITERATION {iteration_num}")
    print(f"==============================================")
    
    df = prepare_data_from_csvs()
    print(f"Total Rows Loaded: {len(df)}")
    
    features = ['rolling_avg_10', 'volume_sum_10']
    X = df[features]
    y = df['target']
    
    # Chronological or random train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"--- EVALUATION METRICS (Iteration {iteration_num}) ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    
    model_filename = f"model_iter{iteration_num}.joblib"
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")
    return acc, prec, rec, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=1, help="1 for v0 only, 2 for v0+v1 merged")
    args = parser.parse_args()
    run_iteration(args.iteration)
