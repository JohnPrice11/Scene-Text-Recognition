#!/usr/bin/env python3
"""
⚡ FAST TrOCR Extraction Script
   - Implements BATCH PROCESSING (Major speedup)
   - Uses FP16 (Half Precision) for faster inference
   - Applies Polygon Masking
"""

import warnings, os, logging
import cv2
import numpy as np
import torch
from shapely.geometry import Polygon
from shapely.ops import unary_union
from tqdm import tqdm
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger().setLevel(logging.ERROR)

# ================= CONFIGURATION =================
MODEL_PATH = "./trocr_finetuned_full" 
RESULTS_DIR = "results_test_polygon"       
BOX_TXT_DIR = "results_test_polygon"    
OUT_TXT_DIR = "result_final_text"
OUT_VIS_DIR = "result_final_vis"

# ⚡ PERFORMANCE SETTINGS
BATCH_SIZE = 16         # Process 16 crops at once (Try 32 if you have 16GB+ VRAM)
USE_FP16 = True         # Use Half Precision (Much faster on GPU)

os.makedirs(OUT_TXT_DIR, exist_ok=True)
os.makedirs(OUT_VIS_DIR, exist_ok=True)

# ================= LOAD MODEL =================
print(" Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(device)
    
    # Force Config
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    if USE_FP16 and device == "cuda":
        model.half() #  Convert to FP16
        print(" FP16 Enabled")
    
    model.eval()
    print(f" Model loaded on {device}")
    
except Exception as e:
    print(f" Error loading model: {e}")
    exit(1)

# ================= HELPER FUNCTIONS =================

def mask_and_crop_polygon(img, poly):
    if poly.is_empty: return None
    try:
        pts = np.array(poly.exterior.coords, dtype=np.int32)
        
        # Masking
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        
        # Cropping
        x, y, w, h = cv2.boundingRect(pts)
        pad = 2
        x = max(0, x - pad); y = max(0, y - pad)
        w = min(img.shape[1] - x, w + 2 * pad)
        h = min(img.shape[0] - y, h + 2 * pad)

        if w < 2 or h < 2: return None
        return masked_img[y:y+h, x:x+w]
    except:
        return None

def batch_inference(crop_list):
    """
    ⚡ Processes a list of images in one go.
    """
    if not crop_list: return []
    
    results = []
    
    # Process in chunks of BATCH_SIZE
    for i in range(0, len(crop_list), BATCH_SIZE):
        batch_crops = crop_list[i : i + BATCH_SIZE]
        
        try:
            # 1. Preprocess Batch
            # Convert all BGR -> RGB and PIL
            pil_batch = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in batch_crops]
            
            pixel_values = processor(images=pil_batch, return_tensors="pt").pixel_values.to(device)
            
            if USE_FP16 and device == "cuda":
                pixel_values = pixel_values.half()

            # 2. Generate Batch
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values,
                    decoder_start_token_id=model.config.decoder_start_token_id,
                    max_length=64,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    num_beams=4 
                )
            
            # 3. Decode Batch
            batch_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            results.extend([text.strip() for text in batch_texts])
            
        except Exception as e:
            print(f" Batch Error: {e}")
            results.extend([""] * len(batch_crops)) # Fill with empty on error
            
    return results

def merge_polygons(polygons, iou_threshold=0.15):
    merged = []
    for poly in polygons:
        merged_flag = False
        for i, existing in enumerate(merged):
            if poly.intersects(existing):
                inter_area = poly.intersection(existing).area
                union_area = poly.union(existing).area
                if union_area > 0 and (inter_area / union_area) > iou_threshold:
                    merged[i] = unary_union([existing, poly])
                    merged_flag = True
                    break
        if not merged_flag:
            merged.append(poly)
    return merged

# ================= MAIN LOOP =================
label_files = sorted([f for f in os.listdir(BOX_TXT_DIR) if f.endswith(".txt")])
print(f" Processing {len(label_files)} files with Batch Size {BATCH_SIZE}...")

for fname in tqdm(label_files):
    base_name = os.path.splitext(fname)[0]
    
    # Find Image
    img_path = os.path.join(RESULTS_DIR, base_name + ".jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(RESULTS_DIR, base_name + ".png")
    if not os.path.exists(img_path): continue

    img = cv2.imread(img_path)
    if img is None: continue

    # 1. Parse ALL Polygons first
    polygons = []
    with open(os.path.join(BOX_TXT_DIR, fname), "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4: continue
            try:
                raw_coords = list(map(float, parts[1:]))
                if len(raw_coords) % 2 != 0: raw_coords = raw_coords[:-1]
                pts = np.array(raw_coords, dtype=np.float32).reshape(-1, 2)
                poly = Polygon(pts)
                if poly.is_valid and poly.area > 5:
                    polygons.append(poly)
            except: continue

    final_polygons = merge_polygons(polygons)
    
    # 2. Prepare Crops for Batching
    valid_crops = []
    valid_polys = [] # Keep track of which poly goes with which crop
    
    for poly in final_polygons:
        crop = mask_and_crop_polygon(img, poly)
        if crop is not None:
            valid_crops.append(crop)
            valid_polys.append(poly)

    # 3. Run Batch Inference (The Speedup!)
    if valid_crops:
        text_results = batch_inference(valid_crops)
        
        # 4. Visualize & Save
        output_lines = []
        for poly, text in zip(valid_polys, text_results):
            if text:
                output_lines.append(text)
                
                # Draw
                pts = np.array(poly.exterior.coords, dtype=np.int32)
                cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                label_pos = (int(pts[:,0].min()), int(pts[:,1].min()) - 5)
                cv2.putText(img, text, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 0), 2)

        # Save Text
        with open(os.path.join(OUT_TXT_DIR, base_name + ".txt"), "w") as f:
            f.write("\n".join(output_lines))
            
        # Save Image
        cv2.imwrite(os.path.join(OUT_VIS_DIR, base_name + ".jpg"), img)

print("\n Finished!")