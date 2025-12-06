#!/usr/bin/env python3
"""
inference_draw_polygons.py (MODIFIED V2)
Runs inference using trained Detectron2 model and saves:
 - Clean .jpg images (with precise, non-merged polygons)
 - .txt files with precise polygon coordinates (x1 y1 x2 y2 x3 y3...)

THIS SCRIPT IS MODIFIED TO:
 - Filter out detections that are contained inside other detections.
 - It discards the *smaller* box when one is inside another, regardless of score.
"""

import os, cv2, torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from tqdm import tqdm
import numpy as np

# --- Paths ---
MODEL_DIR = "output_detectron2"
IMAGES_DIR = "dataset/images"
OUTPUT_DIR = "results_test_polygon_improv" # Use a new, clean output directory

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Config ---
print("Loading Detectron2 model...")
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# *** Using the model path you provided ***
cfg.MODEL.WEIGHTS = os.path.join(MODEL_DIR, "model_0029999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only 'text'
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)
print(f"Model loaded and running on {cfg.MODEL.DEVICE}")

# Create dummy metadata to avoid text labels
metadata = MetadataCatalog.get("text_train")
metadata.thing_classes = ["text"]

# --- Helper function for filtering ---
def calculate_ioa(box1, box2):
    """
    Calculates the Intersection over Area (IoA) of box1 w.r.t box2.
    IoA = area(intersection(box1, box2)) / area(box1)
    This measures how much of box1 is *contained within* box2.
    Boxes are in [x1, y1, x2, y2] format.
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])

    if box1_area == 0:
        return 0

    return intersection_area / box1_area

# --- Run Inference ---
for img_name in tqdm(os.listdir(IMAGES_DIR), desc="Running inference"):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGES_DIR, img_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f" Skipping unreadable: {img_name}")
        continue

    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")

    if len(instances) == 0:
        continue

    # --- *** NEW FILTERING LOGIC (v2) *** ---
    # Get all boxes, scores, and masks
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy() # Keep scores for tie-breaking duplicates
    masks = instances.pred_masks.numpy()

    indices_to_discard = set()
    num_instances = len(instances)

    for i in range(num_instances):
        if i in indices_to_discard: # If already marked, skip
            continue

        for j in range(num_instances):
            if i == j or j in indices_to_discard: # Don't compare to self or already-marked
                continue

            # Check containment both ways
            ioa_i_in_j = calculate_ioa(boxes[i], boxes[j]) # How much of i is in j
            ioa_j_in_i = calculate_ioa(boxes[j], boxes[i]) # How much of j is in i

            # Set a containment threshold (90%)
            containment_threshold = 0.9

            # Case 1: They are near-duplicates (both > 90% contained in each other)
            if ioa_i_in_j > containment_threshold and ioa_j_in_i > containment_threshold:
                # Discard the one with lower confidence score
                if scores[i] < scores[j]:
                    indices_to_discard.add(i)
                    break # i is discarded
                else:
                    indices_to_discard.add(j) # j is discarded

            # Case 2: 'i' is inside 'j' (j is the container)
            # e.g., "WARE" inside "GARAGEWARE" -> discard "WARE" (i)
            elif ioa_i_in_j > containment_threshold:
                indices_to_discard.add(i)
                break # i is discarded

            # Case 3: 'j' is inside 'i' (i is the container)
            elif ioa_j_in_i > containment_threshold:
                indices_to_discard.add(j) # j is discarded

    if len(indices_to_discard) > 0:
        # tqdm.write(f"  Filtering {len(indices_to_discard)} nested boxes for {img_name}")
        pass
    # --- *** END OF NEW FILTERING LOGIC (v2) *** ---


    out_img = image.copy()
    detection_lines_for_txt = []

    # --- Loop over FILTERED instances ---
    for i in range(num_instances):
        # Skip if this index was marked for discarding
        if i in indices_to_discard:
            continue

        mask = masks[i]
        score = scores[i]

        # --- 1. Get Contour from Mask ---
        binary_mask = (mask > 0.5).astype(np.uint8)
        mask_uint8 = binary_mask * 255

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)

        # --- 2. Simplify the contour ---
        peri = cv2.arcLength(contour, True)
        approx_poly = cv2.approxPolyDP(contour, 0.005 * peri, True)

        if len(approx_poly) < 3:
            continue

        # --- 3. Draw the polygon ---
        cv2.polylines(out_img, [approx_poly], isClosed=True, color=(0, 255, 0), thickness=2)

        # --- 4. Save the polygon points ---
        # Flatten the points to 'x1 y1 x2 y2 ...'
        polygon_points = approx_poly.squeeze().flatten().tolist()
        points_str = " ".join(map(str, polygon_points))
        detection_lines_for_txt.append(f"text {points_str} {score:.2f}\n")

    # --- Save visualization ---
    out_img_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(out_img_path, out_img)

    # --- Save .txt file ---
    txt_name = os.path.splitext(img_name)[0] + ".txt"
    txt_path = os.path.join(OUTPUT_DIR, txt_name)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.writelines(detection_lines_for_txt)

print(f"\n Inference completed. Filtered polygon results saved in: {OUTPUT_DIR}")