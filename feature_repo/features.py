from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32
from feast.value_type import ValueType

stock_entity = Entity(
    name="stock_name",
    value_type=ValueType.STRING,
    description="Name of the stock ticker",
)

stock_features_source = FileSource(
    name="stock_features_source",
    path="data/stock_features.parquet",
    timestamp_field="timestamp",
)

stock_rolling_features_view = FeatureView(
    name="stock_rolling_features",
    entities=[stock_entity],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="rolling_avg_10", dtype=Float32),
        Field(name="volume_sum_10", dtype=Float32),
    ],
    online=True,
    source=stock_features_source,
)
