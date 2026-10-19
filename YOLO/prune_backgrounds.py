import os
import shutil
from pathlib import Path

train_img_dir = Path(r"rdd2022\RDD_SPLIT\train\images")
train_lbl_dir = Path(r"rdd2022\RDD_SPLIT\train\labels")
backup_dir = Path(r"rdd2022\RDD_SPLIT\train\backgrounds_backup")
backup_dir.mkdir(exist_ok=True)

keep_limit = 1000  # Maintain ~5% background ratio
pruned = 0
kept = 0

for lbl_path in list(train_lbl_dir.glob("*.txt")):
    # Check if the label file is 0 bytes (empty background)
    if os.path.getsize(lbl_path) == 0:
        if kept < keep_limit:
            kept += 1
            continue
        
        # Locate corresponding image
        img_path = train_img_dir / f"{lbl_path.stem}.jpg"
        if img_path.exists():
            shutil.move(str(img_path), str(backup_dir / img_path.name))
            shutil.move(str(lbl_path), str(backup_dir / lbl_path.name))
            pruned += 1

print(f"Done! Pruned {pruned} background images. Kept {kept} backgrounds.")