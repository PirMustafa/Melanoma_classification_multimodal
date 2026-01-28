"""
Merge ISIC 2018 Dataset into Training Data
"""
import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path(r'E:\Data_Mining_Project\Data')

print('='*60)
print('MERGING ISIC 2018 INTO TRAINING DATA')
print('='*60)

# 1. Load existing train.csv (2019+2020)
train_df = pd.read_csv(data_dir / 'train.csv')
print(f'\n[1] Current train.csv: {len(train_df)} samples')

# 2. Load ISIC 2019 Ground Truth (has 2018 labels)
gt_2019 = pd.read_csv(data_dir / 'ISIC_2019_Training_GroundTruth.csv')
print(f'[2] ISIC 2019 Ground Truth: {len(gt_2019)} samples')

# 3. Load ISIC 2019 Metadata
meta_2019 = pd.read_csv(data_dir / 'ISIC_2019_Training_Metadata.csv')
print(f'[3] ISIC 2019 Metadata: {len(meta_2019)} samples')

# 4. Get ISIC 2018 image list
isic_2018_dir = data_dir / 'ISIC2018_Task3_Training_Input'
isic_2018_images = [f.stem for f in isic_2018_dir.glob('*.jpg')]
print(f'[4] ISIC 2018 images: {len(isic_2018_images)} files')

# 5. Filter to only 2018 images not already in train
existing_images = set(train_df['image_name'].values)
new_2018_images = [img for img in isic_2018_images if img not in existing_images]
print(f'[5] New 2018 images to add: {len(new_2018_images)}')

# 6. Create dataframe for 2018 images
# Get labels from 2019 Ground Truth
gt_2019_filtered = gt_2019[gt_2019['image'].isin(new_2018_images)].copy()
gt_2019_filtered = gt_2019_filtered.rename(columns={'image': 'image_name'})

# Determine diagnosis from one-hot encoded columns
diagnosis_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
gt_2019_filtered['diagnosis'] = gt_2019_filtered[diagnosis_cols].idxmax(axis=1)

# MEL is malignant (target=1), others are benign (target=0)
gt_2019_filtered['target'] = (gt_2019_filtered['MEL'] == 1.0).astype(int)
gt_2019_filtered['benign_malignant'] = gt_2019_filtered['target'].map({0: 'benign', 1: 'malignant'})

# 7. Merge with metadata
meta_2019_filtered = meta_2019[meta_2019['image'].isin(new_2018_images)].copy()
meta_2019_filtered = meta_2019_filtered.rename(columns={'image': 'image_name', 'anatom_site_general': 'anatom_site_general_challenge'})

# Merge labels with metadata
new_df = pd.merge(gt_2019_filtered[['image_name', 'diagnosis', 'target', 'benign_malignant']], 
                  meta_2019_filtered[['image_name', 'sex', 'age_approx', 'anatom_site_general_challenge']],
                  on='image_name', how='left')

# Add patient_id as NaN (not available for 2018)
new_df['patient_id'] = np.nan

# Reorder columns to match train.csv
new_df = new_df[['image_name', 'patient_id', 'sex', 'age_approx', 'anatom_site_general_challenge', 
                  'diagnosis', 'benign_malignant', 'target']]

print(f'[6] Created 2018 dataframe: {len(new_df)} samples')
print(f'    - Malignant: {new_df["target"].sum()}')
print(f'    - Benign: {len(new_df) - new_df["target"].sum()}')

# 8. Merge with existing train.csv
merged_df = pd.concat([train_df, new_df], ignore_index=True)
print(f'\n[7] MERGED DATASET: {len(merged_df)} total samples')
malignant_count = merged_df["target"].sum()
malignant_pct = 100 * merged_df["target"].mean()
print(f'    - Malignant: {malignant_count} ({malignant_pct:.2f}%)')
print(f'    - Benign: {len(merged_df) - malignant_count}')

# 9. Save merged dataset
output_path = data_dir / 'train_with_2018.csv'
merged_df.to_csv(output_path, index=False)
print(f'\n[8] Saved to: {output_path}')

# 10. Show diagnosis distribution
print('\n[9] Diagnosis Distribution in Merged Dataset:')
print(merged_df['diagnosis'].value_counts())

print('\n' + '='*60)
print('MERGE COMPLETE!')
print('='*60)
