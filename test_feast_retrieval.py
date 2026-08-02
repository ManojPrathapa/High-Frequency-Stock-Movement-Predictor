import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")

# Create sample entity dataframe with timestamps to query point-in-time features
df = pd.read_parquet("data/processed_dataset.parquet")
entity_df = df[['timestamp', 'stock_name', 'target']].tail(100).copy()

training_data = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "stock_rolling_features:rolling_avg_10",
        "stock_rolling_features:volume_sum_10",
    ],
).to_df()

print("=== FEAST HISTORICAL FEATURE RETRIEVAL SUCCESSFUL ===")
print(training_data.head(5))
