import pandas as pd

df = pd.read_csv("analysis/selected_features.csv", nrows=5)
print(df.columns.tolist())
