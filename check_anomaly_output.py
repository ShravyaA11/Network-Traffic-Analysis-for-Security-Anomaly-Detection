import pandas as pd

df = pd.read_csv("analysis/anomaly_detection_output.csv", nrows=20)
print(df)
