import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tqdm import tqdm

LOG_FILE = "statistical_analysis.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def statistical_analysis(data_file="preprocess_output.csv", output_folder=None, chunk_size=400000):
    """
    FULL MEMORY-SAFE STATISTICAL ANALYSIS
    ✓ Processes large datasets (10M+ rows)
    ✓ Uses chunking everywhere
    ✓ No concat (prevents RAM crash)
    ✓ Saves stats, missing, correlation, outliers
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if output_folder is None:
        output_folder = os.path.join(script_dir, "analysis")
    os.makedirs(output_folder, exist_ok=True)

    data_path = os.path.join(script_dir, data_file)

    print("⏳ Chunk-loading dataset (memory safe)...")

    # ---------------------------------------------------
    # 1️⃣ READ ONLY FIRST CHUNK FOR BASIC METADATA
    # ---------------------------------------------------
    try:
        first_chunk = next(pd.read_csv(data_path, chunksize=chunk_size, low_memory=False))
        total_columns = list(first_chunk.columns)
        numeric_cols = first_chunk.select_dtypes(include=["float64","int64"]).columns.tolist()

        print("✅ First chunk loaded.")
        logging.info("First chunk loaded.")

    except Exception as e:
        print(f"❌ Error reading file: {e}")
        logging.error(e)
        return

    # ---------------------------------------------------
    # 2️⃣ COUNT TOTAL ROWS USING CHUNKS
    # ---------------------------------------------------
    print("🔢 Counting total rows...")
    total_rows = 0
    for chunk in pd.read_csv(data_path, chunksize=chunk_size, low_memory=False):
        total_rows += len(chunk)

    # SAVE DATASET SUMMARY
    summary_path = os.path.join(output_folder, "dataset_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Total Rows: {total_rows}\n")
        f.write(f"Total Columns: {len(total_columns)}\n")
        f.write("Columns:\n")
        for col in total_columns:
            f.write(f"{col}\n")

    print("📄 Dataset summary saved.")

    # ---------------------------------------------------
    # 3️⃣ MISSING VALUES REPORT (CHUNK-BASED)
    # ---------------------------------------------------
    print("🧹 Computing missing values...")

    missing_counts = {col: 0 for col in total_columns}

    for chunk in pd.read_csv(data_path, chunksize=chunk_size, low_memory=False):
        chunk_missing = chunk.isnull().sum()
        for col in total_columns:
            missing_counts[col] += chunk_missing[col]

    pd.DataFrame({"Column": list(missing_counts.keys()),
                  "Missing": list(missing_counts.values())})\
        .to_csv(os.path.join(output_folder, "missing_values.csv"), index=False)

    print("📄 Missing values report saved.")

    # ---------------------------------------------------
    # 4️⃣ DESCRIPTIVE STATS ONLY FOR FIRST CHUNK
    # ---------------------------------------------------
    first_chunk.describe(include="all")\
       .to_csv(os.path.join(output_folder, "descriptive_stats.csv"))

    print("📄 Descriptive stats saved (first chunk).")

    # ---------------------------------------------------
    # 5️⃣ ATTACK TYPE DISTRIBUTION (CHUNKED)
    # ---------------------------------------------------
    print("📊 Counting attack types...")

    attack_counts = {}

    for chunk in pd.read_csv(data_path, chunksize=chunk_size, low_memory=False):
        if "attack_type" in chunk.columns:
            vals = chunk["attack_type"].value_counts()
            for k, v in vals.items():
                attack_counts[k] = attack_counts.get(k, 0) + v

    attack_df = pd.DataFrame.from_dict(attack_counts, orient="index", columns=["count"])
    attack_df.to_csv(os.path.join(output_folder, "attack_type_distribution.csv"))

    print("📄 Attack type distribution saved.")

    # ---------------------------------------------------
    # 6️⃣ CORRELATION HEATMAP (SAMPLED TO SAVE RAM)
    # ---------------------------------------------------
    print("🔍 Building sampled correlation heatmap...")

    sample = pd.read_csv(data_path, nrows=200000, low_memory=False)
    numeric_sample = sample.select_dtypes(include=["float64","int64"])

    if len(numeric_sample.columns) > 1:
        plt.figure(figsize=(14, 7), dpi=200)
        sns.heatmap(numeric_sample.corr(), cmap="coolwarm")
        plt.title("Correlation Heatmap (Sampled Data)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "correlation_heatmap.png"))
        plt.close()

    print("📊 Correlation heatmap saved.")

    # ---------------------------------------------------
    # 7️⃣ OUTLIER DETECTION (SAMPLED)
    # ---------------------------------------------------
    print("🔍 Detecting outliers (sample)...")

    Q1 = numeric_sample.quantile(0.25)
    Q3 = numeric_sample.quantile(0.75)
    IQR = Q3 - Q1

    outliers = ((numeric_sample < (Q1 - 1.5 * IQR)) |
                (numeric_sample > (Q3 + 1.5 * IQR))).sum()

    pd.DataFrame({"Feature": outliers.index,
                  "Outlier_Count": outliers.values})\
        .to_csv(os.path.join(output_folder, "outlier_summary.csv"), index=False)

    print("📌 Outlier summary saved.")

    print("\n🎉 ALL STATISTICAL ANALYSIS COMPLETED SUCCESSFULLY!")
    print("📁 Results saved in:", output_folder)


# ---------------------------------------------------
# RUN
# ---------------------------------------------------
if __name__ == "__main__":
    statistical_analysis("preprocess_output.csv")
