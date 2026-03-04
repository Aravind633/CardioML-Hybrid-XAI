import pandas as pd

# Load MIT-BIH test set
df = pd.read_csv("data/mitbih_test.csv", header=None)

# Take ONE heartbeat (row 0)
ecg_beat = df.iloc[0, :-1]  # drop label

# Save as CSV (187 values)
ecg_beat.to_csv("sample_ecg.csv", index=False, header=False)

print("sample_ecg.csv created with", len(ecg_beat), "values")
