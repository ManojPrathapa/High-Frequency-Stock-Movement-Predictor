import os
import glob
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Configure local SQLite backend so Model Registry works properly
db_path = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(db_path)
experiment_name = "Stock_Movement_Tuning"
mlflow.set_experiment(experiment_name)

def load_data():
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
        
        df['rolling_avg_10'] = df['close'].rolling(window='10min', min_periods=1).mean()
        df['volume_sum_10'] = df['volume'].rolling(window='10min', min_periods=1).sum()
        
        df['close_5min_future'] = df['close'].shift(-5)
        df['target'] = (df['close_5min_future'] > df['close']).astype(int)
        
        df.dropna(subset=['rolling_avg_10', 'volume_sum_10', 'target'], inplace=True)
        df.drop(columns=['close_5min_future'], inplace=True)
        df.reset_index(inplace=True)
        all_dfs.append(df)
        
    return pd.concat(all_dfs, ignore_index=True)

def run_sweep():
    df = load_data()
    features = ['rolling_avg_10', 'volume_sum_10']
    X = df[features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Define sweep grid
    param_grid = [
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 50, "max_depth": 10},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 100, "max_depth": 15},
    ]
    
    best_f1 = -1.0
    best_run_id = None
    
    print("Starting Hyperparameter Tuning Sweep...")
    for params in param_grid:
        with mlflow.start_run() as run:
            n_est = params["n_estimators"]
            depth = params["max_depth"]
            
            model = RandomForestClassifier(
                n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Log Parameters and Metrics
            mlflow.log_param("n_estimators", n_est)
            mlflow.log_param("max_depth", depth)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            
            # Log Model Artifact
            mlflow.sklearn.log_model(model, "model")
            
            print(f"Run {run.info.run_id} | n_est: {n_est}, depth: {depth} | f1: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_run_id = run.info.run_id
                
    print(f"\n==============================================")
    print(f"   BEST RUN SELECTED: {best_run_id}")
    print(f"   BEST F1 SCORE    : {best_f1:.4f}")
    print(f"==============================================")
    
    # Register best model to Model Registry
    model_uri = f"runs:/{best_run_id}/model"
    registered_model_name = "stock_movement_predictor"
    print(f"Registering best model to registry as '{registered_model_name}'...")
    mlflow.register_model(model_uri, registered_model_name)
    print("Model Registration Complete!")

if __name__ == "__main__":
    run_sweep()
