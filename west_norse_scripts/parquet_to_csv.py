import pandas as pd
df = pd.read_parquet('merged_ner_dataset.parquet')
df.to_csv('merged_ner_dataset.csv', index = False)