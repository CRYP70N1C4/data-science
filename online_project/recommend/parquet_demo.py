import pandas as pd



df = pd.read_parquet('../../_dataset/recommend/xgame/data.parquet', engine='pyarrow')

print(df.head(10))