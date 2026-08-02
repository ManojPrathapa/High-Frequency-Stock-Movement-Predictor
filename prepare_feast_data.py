import glob
import os
import pandas as pd
import numpy as np

csv_files = glob.glob("data/*.csv")
print(f"Found CSV files: {csv_files}")

all_dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    # Extract ticker name from filename if column is missing or to standardize
    ticker = os.path.basename(file).split('__')[0]
    df['stock_name'] = ticker
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.set_index('timestamp', inplace=True)
    df.ffill(inplace=True)
    
    # Compute rolling features per stock
    df['rolling_avg_10'] = df['close'].rolling(window='10min', min_periods=1).mean()
    df['volume_sum_10'] = df['volume'].rolling(window='10min', min_periods=1).sum()
    
    # Compute Target: 1 if price goes up 5 mins later
    df['close_5min_future'] = df['close'].shift(-5)
    df['target'] = (df['close_5min_future'] > df['close']).astype(int)
    
    df.dropna(subset=['rolling_avg_10', 'volume_sum_10', 'target'], inplace=True)
    df.drop(columns=['close_5min_future'], inplace=True)
    df.reset_index(inplace=True)
    all_dfs.append(df)

full_df = pd.concat(all_dfs, ignore_index=True)
full_df['rolling_avg_10'] = full_df['rolling_avg_10'].astype('float32')
full_df['volume_sum_10'] = full_df['volume_sum_10'].astype('float32')

# Ensure Feast output directory exists
os.makedirs("feature_repo/data", exist_ok=True)
parquet_path = "feature_repo/data/stock_features.parquet"
full_df.to_parquet(parquet_path, index=False)
print(f"Successfully saved {len(full_df)} rows to {parquet_path}")

# Save full processed dataframe for iterative training
full_df.to_parquet("data/processed_dataset.parquet", index=False)
