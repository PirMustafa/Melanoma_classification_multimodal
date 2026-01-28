import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# --- 1. CONFIGURATION ---
# Base directory for your E: drive project
DATA_DIR = r"E:\Data_Mining_Project\Data"

# Output file path
OUTPUT_PATH = os.path.join(DATA_DIR, "train_merged_master.csv")

def main():
    print("🚀 Starting Data Merger for ISIC 2019 + 2020...")

    # --- 2. PROCESS 2020 DATA ---
    print("📂 Processing 2020 metadata...")
    df_2020 = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_2020["year"] = 2020
    # Map the relative path to the full local path for your RTX 5060
    df_2020["filepath"] = df_2020["image_name"].apply(
        lambda x: os.path.join(DATA_DIR, "jpeg", "train", f"{x}.jpg")
    )

    # --- 3. PROCESS 2019 DATA ---
    # Based on your folder structure: AK, BCC, BKL, DF, MEL, NV, SCC, VASC
    print("📂 Processing 2019 folders...")
    folders_2019 = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
    data_2019 = []

    for folder in folders_2019:
        folder_path = os.path.join(DATA_DIR, folder)
        # ONLY 'MEL' (Melanoma) is target 1. Everything else is target 0.
        target = 1 if folder == "MEL" else 0
        
        if os.path.exists(folder_path):
            images = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
            for img in tqdm(images, desc=f"Indexing {folder}", leave=False):
                data_2019.append({
                    "image_name": img.replace(".jpg", ""),
                    "target": target,
                    "year": 2019,
                    "filepath": os.path.join(folder_path, img)
                })
        else:
            print(f"⚠️ Warning: Folder {folder} not found at {folder_path}")

    df_2019 = pd.DataFrame(data_2019)

    # --- 4. MERGE & STABILIZE ---
    # We only keep the core columns to keep the dataframe lightweight
    cols = ["image_name", "target", "year", "filepath"]
    train_all = pd.concat([df_2020[cols], df_2019[cols]], axis=0).reset_index(drop=True)

    # --- 5. THE WINNER'S STRATIFIED FOLDING ---
    # We stratify by BOTH Year and Target to prevent local score 'shake'
    print("🎲 Generating Stratified 5-Fold split...")
    train_all["stratify_group"] = (
        train_all["year"].astype(str) + "_" + train_all["target"].astype(str)
    )
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_all["fold"] = -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_all, train_all["stratify_group"])):
        train_all.loc[val_idx, "fold"] = fold

    # --- 6. SAVE & REPORT ---
    train_all.drop(columns=["stratify_group"], inplace=True)
    train_all.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "="*30)
    print(f"✅ Master File Saved: {OUTPUT_PATH}")
    print(f"📊 Total Samples: {len(train_all)}")
    print(f"🔬 Malignant Cases: {train_all['target'].sum()} (from 584 up to ~5,000!)")
    print("="*30)

if __name__ == "__main__":
    main()