import os
import glob
import pandas as pd
from tqdm import tqdm
import logging

# -----------------------------------------------------------
#  CONFIGURE LOGGING
# -----------------------------------------------------------
LOG_FILE = "preprocess.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def preprocess_data(data_folder="data", output_file=None, chunksize=100000):
    """
    FULL RAM-SAFE PREPROCESSING for massive Web-IDS23 datasets.
    FIXES:
      ✓ No loading all chunks into RAM
      ✓ Writes directly to output CSV
      ✓ No concat (which caused 2.7GB crash)
      ✓ Correct attack_type extraction
    """

    # -----------------------------------------------------------
    # 1. Output file name
    # -----------------------------------------------------------
    if output_file is None:
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        output_file = f"{script_name}_output.csv"

    output_path = os.path.join(os.path.dirname(__file__), output_file)

    # -----------------------------------------------------------
    # 2. Remove old output if exists
    # -----------------------------------------------------------
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except:
            print("❌ Close the output file (Excel) and run again.")
            return

    # -----------------------------------------------------------
    # 3. Find CSV files in folder
    # -----------------------------------------------------------
    files = glob.glob(os.path.join(data_folder, "*.csv"))
    print(f"📂 Total CSV files found: {len(files)}")

    if not files:
        print("❌ No CSV files found.")
        return

    print("\n🚀 Starting FULL DATA PREPROCESSING...\n")

    # -----------------------------------------------------------
    # 4. Process each file
    # -----------------------------------------------------------
    for file in files:
        print(f"\n📄 Processing file: {os.path.basename(file)}")
        logging.info(f"Processing file: {file}")

        # Detect attack type
        fname = os.path.basename(file).lower()

        if "benign" in fname:
            attack_type = "benign"
        else:
            name = os.path.splitext(fname)[0]
            name = (
                name.replace("web_ids23", "")
                .replace("webids23", "")
                .replace("web", "")
                .replace("__", "_")
                .strip("_")
            )
            attack_type = name.split("_", 1)[-1]

        try:
            # STREAMING CHUNK READER
            for chunk in tqdm(
                pd.read_csv(file, chunksize=chunksize, on_bad_lines="skip", low_memory=False),
                desc=f"Reading {os.path.basename(file)}",
                unit="chunk"
            ):
                # Clean column names
                chunk.columns = [
                    str(c).strip().lower().replace(" ", "_")
                    for c in chunk.columns
                ]

                # Assign label
                chunk["attack_type"] = attack_type

                # Fill numeric missing values
                numeric_cols = chunk.select_dtypes(include=["int64", "float64"]).columns
                for col in numeric_cols:
                    chunk[col] = chunk[col].fillna(chunk[col].mean())

                # Fill categorical missing values
                categorical_cols = chunk.select_dtypes(include=["object"]).columns
                for col in categorical_cols:
                    chunk[col] = chunk[col].fillna(chunk[col].mode()[0])

                # Remove duplicates inside chunk
                chunk = chunk.drop_duplicates()

                # -----------------------------------------------------------
                # RAM-SAFE STREAMING WRITE (MAIN FIX)
                # -----------------------------------------------------------
                chunk.to_csv(
                    output_path,
                    mode="a",
                    index=False,
                    header=not os.path.exists(output_path)
                )

        except Exception as e:
            print(f"⚠ Failed: {file} → {e}")
            logging.error(f"Error processing {file}: {e}")
            continue

    print("\n✅ PREPROCESSING COMPLETE!")
    print(f"💾 Saved as: {output_path}")
    print(f"📘 Log saved at: {LOG_FILE}")


# -----------------------------------------------------------
# RUN
# -----------------------------------------------------------
if __name__ == "__main__":
    preprocess_data(data_folder=r"C:\Users\major project\data")
