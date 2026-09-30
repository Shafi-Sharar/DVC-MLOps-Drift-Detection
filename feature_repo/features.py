from datetime import timedelta
import pandas as pd
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

file_source = FileSource(name="processed_data_source", path="../processed_data.csv", timestamp_field="event_timestamp",)

user_entity = Entity(name="user_id", description="Primary Key for Entity")

data_feature_view = FeatureView(name="user_feature_view", entities=[user_entity], ttl=timedelta(days=1), schema=[Field(name="feature_1", dtype=Float32), Field(name="feature_2", dtype=Float32),], online=True, source=file_source,)