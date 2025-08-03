import os
import csv

# ✅ Set your dataset root
dataset_dir = r"C:\Brain MRI Seg\lgg-mri-segmentation\kaggle_3m"
output_csv = os.path.join(dataset_dir, "test_data.csv")

# ✅ Extensions
image_ext = ".tif"
mask_ext = "_mask.tif"

# ✅ Collect all valid image-mask pairs
pairs = []

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(image_ext) and not file.endswith(mask_ext):
            image_path = os.path.join(root, file)
            mask_path = os.path.join(root, file.replace(".tif", "_mask.tif"))

            if os.path.exists(mask_path):
                pairs.append((image_path, mask_path))
            else:
                print(f"⚠️ Skipping {file} - mask not found")

# ✅ Write to CSV
if pairs:
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_filename", "mask_filename"])
        writer.writerows(pairs)
    print(f"✅ test_data.csv created with {len(pairs)} valid image-mask pairs")
else:
    print("❌ No valid image-mask pairs found!")
