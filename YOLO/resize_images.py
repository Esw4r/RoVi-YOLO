from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

folders = [
    Path(r"rdd2022\RDD_SPLIT\train\images"),
    Path(r"rdd2022\RDD_SPLIT\val\images")
]

def process_image(img_path):
    try:
        with Image.open(img_path) as img:
            if max(img.size) > 640:
                img.thumbnail((640, 640), Image.Resampling.LANCZOS)
                img.save(img_path, "JPEG", quality=85)
    except Exception as e:
        print(f"Error {img_path}: {e}")

for folder in folders:
    print(f"Resizing images in {folder}...")
    img_list = list(folder.glob("*.jpg"))
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(process_image, img_list)

print("All images resized to 640px.")