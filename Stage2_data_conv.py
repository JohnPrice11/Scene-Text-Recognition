import json
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# CONFIGURATION
JSON_FILE = "dataset/coco_annotations_fixed/train_coco.json"
IMAGE_DIR = "dataset/images"           
OUTPUT_DIR = "train_data/crops"
METADATA_FILE = "train_data/metadata.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def prepare_data():
    print(" Loading COCO JSON...")
    with open(JSON_FILE, 'r') as f:
        coco = json.load(f)

    # Create a map of image_id -> file_name
    images_map = {img['id']: img['file_name'] for img in coco['images']}
    
    data_rows = []
    print(f" Processing {len(coco['annotations'])} annotations...")

    for ann in tqdm(coco['annotations']):
        # 1. Basic Validation
        img_id = ann['image_id']
        if img_id not in images_map: continue
        
        # Get text from attributes (Handling your specific JSON structure)
        text_label = ann.get('attributes', {}).get('text', "")
        
        # Skip illegible or empty text
        if not text_label or text_label == "###":
            continue

        # 2. Load Image
        file_name = images_map[img_id]
        img_path = os.path.join(IMAGE_DIR, file_name)
        
        if not os.path.exists(img_path):
            # Try appending .jpg if missing (common dataset issue)
            continue

        original_img = cv2.imread(img_path)
        if original_img is None: continue

        # 3. Handle Segmentation (Polygon)
        # COCO segmentation is a list of lists: [[x1, y1, x2, y2...]]
        if 'segmentation' in ann and len(ann['segmentation']) > 0:
            seg_points = ann['segmentation'][0]
            poly = np.array(seg_points).reshape(-1, 2).astype(np.int32)
            
            # --- ADVANCED CROPPING STRATEGY ---
            # Create a mask for the polygon
            mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            
            # Apply mask to image (Black out background)
            # This helps the model focus ONLY on the text curve
            masked_img = cv2.bitwise_and(original_img, original_img, mask=mask)
            
            # Crop the bounding rect of the polygon
            x, y, w, h = cv2.boundingRect(poly)
            
            # Add slight padding
            pad = 2
            h_img, w_img = original_img.shape[:2]
            x = max(0, x - pad); y = max(0, y - pad)
            w = min(w_img - x, w + 2*pad); h = min(h_img - y, h + 2*pad)
            
            crop = masked_img[y:y+h, x:x+w]
            
            # If crop is too small/empty, skip
            if crop.size == 0 or w < 5 or h < 5: continue
            
            # Save Crop
            crop_name = f"crop_{ann['id']}.jpg"
            save_path = os.path.join(OUTPUT_DIR, crop_name)
            cv2.imwrite(save_path, crop)
            
            data_rows.append({"file_name": crop_name, "text": text_label})

    # Save Metadata for Training
    df = pd.DataFrame(data_rows)
    df.to_csv(METADATA_FILE, index=False)
    print(f"\n Saved {len(df)} text crops to {OUTPUT_DIR}")

if __name__ == "__main__":
    prepare_data()