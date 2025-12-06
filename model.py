import os, cv2, torch
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import torch
import warnings
import logging
from shapely.geometry import Polygon
from shapely.ops import unary_union
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def _predict_bounding_box(image_path, output_folder):
    """
    EXACT same output as your original script but for ONE IMAGE.

    Inputs:
        image_path (str)     : path to input image
        output_folder (str)  : folder to save .jpg and .txt

    Returns:
        (output_image_path, output_txt_path)
    """

    # -------------------------
    # Load Detectron2 model
    # -------------------------
    MODEL_DIR = "output_detectron2"      # same as your code
    os.makedirs(output_folder, exist_ok=True)

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = os.path.join(MODEL_DIR, "model_0029999.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = DefaultPredictor(cfg)
    MetadataCatalog.get("text_train").thing_classes = ["text"]

    # -------------------------
    # Read the image
    # -------------------------
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Unreadable image: " + image_path)

    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")

    if len(instances) == 0:
        # Create empty txt file
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_txt = os.path.join(output_folder, base + ".txt")
        open(out_txt, "w").close()
        return None, out_txt

    boxes = instances.pred_boxes.tensor.numpy()
    masks = instances.pred_masks.numpy()
    scores = instances.scores.numpy()
    num = len(instances)
    discard = set()

    # -------------------------
    # Filtering logic (same as your code)
    # -------------------------
    def calculate_ioa(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        iw = max(0, x2 - x1)
        ih = max(0, y2 - y1)
        inter = iw * ih

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        if area1 == 0:
            return 0
        return inter / area1

    for i in range(num):
        if i in discard:
            continue

        for j in range(num):
            if i == j or j in discard:
                continue

            ioa_i_j = calculate_ioa(boxes[i], boxes[j])
            ioa_j_i = calculate_ioa(boxes[j], boxes[i])
            t = 0.9

            if ioa_i_j > t and ioa_j_i > t:
                if scores[i] < scores[j]:
                    discard.add(i)
                    break
                else:
                    discard.add(j)

            elif ioa_i_j > t:
                discard.add(i)
                break

            elif ioa_j_i > t:
                discard.add(j)

    # -------------------------
    # Save results
    # -------------------------
    out_img = image.copy()
    txt_lines = []

    for i in range(num):
        if i in discard:
            continue

        mask = masks[i]
        score = scores[i]

        binary = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.005 * peri, True)
        if len(approx) < 3:
            continue

        cv2.polylines(out_img, [approx], True, (0, 255, 0), 2)

        pts = approx.squeeze().flatten().tolist()
        pts_str = " ".join(str(v) for v in pts)
        txt_lines.append(f"text {pts_str} {score:.2f}\n")

    # -------------------------
    # Write output files
    # -------------------------
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_image_path = os.path.join(output_folder, base + "@"+".jpg")
    out_txt_path   = os.path.join(output_folder, base + ".txt")

    cv2.imwrite(out_image_path, out_img)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.writelines(txt_lines)

    return out_image_path, out_txt_path

def _predict_text(img_path, txt_path):
    
    # ================= CONFIGURATION =================
    MODEL_PATH = "./trocr_finetuned_full" 
    OUTPUT_DIR = "webdata"
    BATCH_SIZE = 16      
    USE_FP16 = True      

    # Suppress warnings
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.getLogger().setLevel(logging.ERROR)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ================= LOAD MODEL =================
    # Using global to avoid reloading if function is called multiple times in one session
    global model, processor, device
    
    if 'model' not in globals():
        print("⏳ Loading model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            
            if os.path.exists(MODEL_PATH):
                model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(device)
            else:
                model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(device)
            
            # Force Config
            model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
            model.config.pad_token_id = processor.tokenizer.pad_token_id
            model.config.vocab_size = model.config.decoder.vocab_size
            
            if USE_FP16 and device == "cuda":
                model.half() # ⚡ Convert to FP16
                print("⚡ FP16 Enabled")
            
            model.eval()
            print(f"✅ Model loaded on {device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return

    # ================= HELPER FUNCTIONS (NESTED) =================

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
        if not crop_list: return []
        results = []
        
        # Process in chunks of BATCH_SIZE
        for i in range(0, len(crop_list), BATCH_SIZE):
            batch_crops = crop_list[i : i + BATCH_SIZE]
            
            try:
                # 1. Preprocess Batch
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
                print(f"⚠️ Batch Error: {e}")
                results.extend([""] * len(batch_crops)) 
                
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

    # ================= MAIN LOGIC =================
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    print(f"🚀 Processing {base_name}...")

    # Load Image
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return
    img = cv2.imread(img_path)
    if img is None: return

    # 1. Parse Polygons
    polygons = []
    if not os.path.exists(txt_path):
        print(f"❌ Text path not found: {txt_path}")
        return

    with open(txt_path, "r") as f:
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
    
    # 2. Prepare Crops
    valid_crops = []
    valid_polys = [] 
    
    for poly in final_polygons:
        crop = mask_and_crop_polygon(img, poly)
        if crop is not None:
            valid_crops.append(crop)
            valid_polys.append(poly)

    # 3. Run Inference
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
        with open(os.path.join(OUTPUT_DIR, base_name + "final"+".txt"), "w") as f:
            f.write("\n".join(output_lines))
            
        # Save Image
        cv2.imwrite(os.path.join(OUTPUT_DIR, base_name + "final"+".jpg"), img)
        print(f"✅ Output saved in {OUTPUT_DIR}")
    else:
        print("⚠️ No valid crops found.")

# def predict_whole():
#     folder = "./webdata"
#     full_path = ""
#     for file in os.listdir(folder):
#         if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
#             full_path = os.path.join(folder, file)
#             print(full_path)
#     _predict_bounding_box(full_path, "./webdata")
#     full_path_txt = full_path[:-4]+".txt"
#     _predict_text(full_path, full_path_txt)
# predict_whole()

# def predict_whole(image_path):
#     # Call bounding box function
#     _predict_bounding_box(image_path, "./webdata")

#     # Replace last 4 chars with .txt
#     full_path_txt = image_path[:-4] + ".txt"

#     # Call text prediction
#     _predict_text(image_path, full_path_txt)

#     return image_path[:-4]+"@.txt", image_path[:-4]+"_final.jpg"

import base64

def predict_whole(image_path):
    # 1. Run bounding box model into the `webdata` folder
    output_folder = "./webdata"
    _predict_bounding_box(image_path, output_folder)

    base, _ = os.path.splitext(os.path.basename(image_path))

    # 2. OCR coordinates text file path (written by _predict_bounding_box)
    txt_coordinates_path = os.path.join(output_folder, base + ".txt")

    # 3. Run text OCR (writes final text into webdata/<base>final.txt)
    _predict_text(image_path, txt_coordinates_path)

    # 4. Read the predicted text file & convert into array
    txt_predicted_path = os.path.join(output_folder, base + "final.txt")
    detections = []
    if os.path.exists(txt_predicted_path):
        try:
            with open(txt_predicted_path, "r", encoding="utf-8") as f:
                detections = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            print(f"Error reading predicted text: {e}")
            detections = []
    else:
        print(f"Predicted text file not found: {txt_predicted_path}")

    # 5. Final annotated bounding-box image path (written by _predict_text)
    final_img_path = os.path.join(output_folder, base + "final.jpg")

    # 6. Convert final image into base64 string (fallback to original image)
    final_image_bytes = ""
    img_for_encoding_path = final_img_path if os.path.exists(final_img_path) else image_path
    if img_for_encoding_path and os.path.exists(img_for_encoding_path):
        try:
            with open(img_for_encoding_path, "rb") as img_file:
                final_image_bytes = base64.b64encode(img_file.read()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding image: {e}")
            final_image_bytes = ""
    else:
        print(f"No image found to encode (checked {final_img_path} and {image_path})")

    # 7. Cleanup generated files in the output folder
    candidates = [
        image_path,
        txt_predicted_path,
        os.path.join(output_folder, base + "@.jpg"),
        final_img_path,
        txt_coordinates_path,
    ]
    for p in candidates:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    return detections, final_image_bytes
