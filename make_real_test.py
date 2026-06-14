import pandas as pd
import os

# ================================
# PATHS
# ================================
BASE = "C:/Users/major project/data"   # <-- your real data folder path

attack_files = [
    os.path.join(BASE, "web-ids23_bruteforce_http.csv"),
    os.path.join(BASE, "web-ids23_dos_http.csv"),
    os.path.join(BASE, "web-ids23_portscan.csv"),
    os.path.join(BASE, "web-ids23_sql_injection_http.csv"),
    os.path.join(BASE, "web-ids23_xss_http.csv"),
    os.path.join(BASE, "web-ids23_ssrf_http.csv"),
    os.path.join(BASE, "web-ids23_revshell_http.csv"),
]

benign_file = os.path.join(BASE, "web-ids23_benign.csv")

# ================================
# BUILD TEST DATASET
# ================================
frames = []

# BENIGN → 100 rows
ben = pd.read_csv(benign_file)
ben_sample = ben.sample(100, random_state=42)
frames.append(ben_sample)

# ATTACKS → 50 rows from EACH attack file
for f in attack_files:
    df = pd.read_csv(f)
    df_sample = df.sample(50, random_state=42)
    frames.append(df_sample)

# ================================
# CONCAT AND SAVE
# ================================
final = pd.concat(frames, ignore_index=True)
final.to_csv("real_test.csv", index=False)

print("🎉 real_test.csv CREATED SUCCESSFULLY!")
print("Total rows:", len(final))
print("\nAttack Types Count:")
print(final["attack_type"].value_counts())
