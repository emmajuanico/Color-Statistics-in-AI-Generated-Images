import os
import sys
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_closing, binary_fill_holes

sys.path.append("/Grounded-Segment-Anything/GroundingDINO")
from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import sam_model_registry, SamPredictor


MODE_PROVA = False 
MAX_IMATGES_PROVA = 2 #nmoes segmentem dues

BASE_INPUT = "/imatges_reals/imatges"
BASE_OUTPUT = "/imatges_reals/segmentades"
BASE_DESCARTATS = "{}/descartats".format(BASE_OUTPUT)

TARGET_OBJECTS = ["bucket", "pen", "coffee mug", "chair"]

DINO_CONFIG = "/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_CHECKPOINT = "/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT = "/sam_vit_h_4b8939.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

# config per objecte (només thresholds i expansió bbox)
OBJECT_CONFIG = {
    "bucket": {
        "box_threshold": 0.35,
        "text_threshold": 0.28,
        "bbox_expansion": 0.25,
    },
    "coffee mug": {
        "box_threshold": 0.35,
        "text_threshold": 0.28,
        "bbox_expansion": 0.15,
    },
    "chair": {
        "box_threshold": 0.32,
        "text_threshold": 0.25,
        "bbox_expansion": 0.10,
    },
    "pen": {
        "box_threshold": 0.35,
        "text_threshold": 0.30,
        "bbox_expansion": 0.05,
    }
}

model_dino = load_model(DINO_CONFIG, DINO_CHECKPOINT)
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(device)
predictor = SamPredictor(sam)


def omplir_forats_interiors(mascara):
    """omple forats interiors excepte els 2 més grans"""
    invertida = ~mascara
    labeled, num = label(invertida)
    if num <= 1:
        return mascara
    bincount = np.bincount(labeled.ravel())
    mides = [(feat_idx, bincount[feat_idx]) for feat_idx in range(1, num + 1)]
    mides_ordenades = sorted(mides, key=lambda x: x[1], reverse=True)
    
    no_omplir = set()
    no_omplir.add(mides_ordenades[0][0])
    if len(mides_ordenades) > 1:
        no_omplir.add(mides_ordenades[1][0])
    
    mascara_plena = mascara.copy()
    for feat_idx, mida in mides:
        if feat_idx not in no_omplir:
            mascara_plena[labeled == feat_idx] = True
    return mascara_plena


#bucle
stats = {'processades': 0, 'ok': 0, 'descartades': 0, 'errors': 0}

for object_type in os.listdir(BASE_INPUT):
    if object_type not in TARGET_OBJECTS:
        continue
    obj_in_path = os.path.join(BASE_INPUT, object_type)
    if not os.path.isdir(obj_in_path):
        continue
    
    config = OBJECT_CONFIG.get(object_type, OBJECT_CONFIG["pen"])
    
    for color in os.listdir(obj_in_path):
        color_in_path = os.path.join(obj_in_path, color)
        if not os.path.isdir(color_in_path):
            continue
        
        out_obj_dir  = "{}/{}/{}/objecte".format(BASE_OUTPUT, object_type, color)
        out_fons_dir = "{}/{}/{}/fons".format(BASE_OUTPUT, object_type, color)
        out_desc_dir = "{}/{}/{}".format(BASE_DESCARTATS, object_type, color)
        os.makedirs(out_obj_dir,  exist_ok=True)
        os.makedirs(out_fons_dir, exist_ok=True)
        os.makedirs(out_desc_dir, exist_ok=True)
        
        imatges = sorted([f for f in os.listdir(color_in_path) if f.endswith(".png")])
        
        if MODE_PROVA:
            imatges = imatges[:MAX_IMATGES_PROVA]
        
        
        for nom_img in imatges:
            img_path  = os.path.join(color_in_path, nom_img)
            save_path = os.path.join(out_obj_dir, nom_img)
            
            # saltar si ja existeix
            if os.path.exists(save_path):
                continue
            
            stats['processades'] += 1
            
            try:
                if object_type == "coffee mug":
                    TEXT_PROMPT = "the {} {} with the coffee inside".format(color, object_type)
                else:
                    TEXT_PROMPT = "the {} {}".format(color, object_type)
                
                image_source, image_transformed = load_image(img_path)
                boxes, logits, phrases = predict(
                    model=model_dino, 
                    image=image_transformed, 
                    caption=TEXT_PROMPT,
                    box_threshold=config["box_threshold"],
                    text_threshold=config["text_threshold"],
                    device=device
                )
                
                if len(boxes) == 0:
                    Image.fromarray(image_source).save(os.path.join(out_desc_dir, nom_img))
                    print("   DESCARTAT: {} (cap deteccio)".format(nom_img))
                    stats['descartades'] += 1
                    continue
                
                predictor.set_image(image_source)
                h, w, _ = image_source.shape
                mascara_final = np.zeros((h, w), dtype=bool)
                
                for i in range(len(boxes)):
                    box = boxes[i] * torch.Tensor([w, h, w, h])
                    cx, cy = box[0].item(), box[1].item()
                    bw, bh = box[2].item(), box[3].item()
                    
                    expansion = bw * config["bbox_expansion"]
                    
                    input_box = np.array([
                        max(0, cx - bw/2 - expansion), 
                        max(0, cy - bh/2 - expansion),
                        min(w, cx + bw/2 + expansion), 
                        min(h, cy + bh/2 + expansion)
                    ])
                    
                    if object_type == "coffee mug":
                        point_coords = np.array([
                            [cx, cy],
                            [cx, cy - bh * 0.15],
                            [cx - bw * 0.35, cy],
                        ])
                        point_labels = np.array([1, 1, 0])
                    else:
                        point_coords = np.array([[cx, cy]])
                        point_labels = np.array([1])
                    
                    masks, scores, _ = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=input_box,
                        multimask_output=True
                    )
                    
                    idx_valids = [j for j, m in enumerate(masks) if m.sum() < (h * w * 0.75)]
                    best_mask = masks[idx_valids[np.argmax(scores[idx_valids])]] if idx_valids else masks[np.argmax(scores)]
                    mascara_final = np.logical_or(mascara_final, best_mask)
                
                mascara_final = binary_closing(mascara_final, structure=np.ones((3, 3)))
                
                if object_type in ["coffee mug", "bucket"]:
                    mascara_final = omplir_forats_interiors(mascara_final)
                
                labeled_array, num_features = label(mascara_final)
                if num_features > 1:
                    bincount = np.bincount(labeled_array.ravel())
                    mida_maxima = np.max(bincount[1:])
                    for feat_idx in range(1, num_features + 1):
                        if bincount[feat_idx] < (mida_maxima * 0.1) and bincount[feat_idx] < 600:
                            mascara_final[labeled_array == feat_idx] = False
                
                if mascara_final.sum() == 0:
                    Image.fromarray(image_source).save(os.path.join(out_desc_dir, nom_img))
                    print("   DESCARTAT: {} (mascara buida)".format(nom_img))
                    stats['descartades'] += 1
                    continue
                
                # inversio
                cx_check = int(boxes[0][0].item() * w)
                cy_check = int(boxes[0][1].item() * h)
                cx_check = max(0, min(w-1, cx_check))
                cy_check = max(0, min(h-1, cy_check))
                if not mascara_final[cy_check, cx_check]:
                    print(" Màscara invertida (centre no dins màscara), corregint...".format(nom_img))
                    mascara_final = ~mascara_final
                
                img_rgba = Image.fromarray(image_source).convert("RGBA")
                
                arr_obj = np.array(img_rgba)
                arr_obj[~mascara_final, 3] = 0
                Image.fromarray(arr_obj).save(save_path)
                
                arr_fons = np.array(img_rgba)
                arr_fons[mascara_final, 3] = 0
                Image.fromarray(arr_fons).save(os.path.join(out_fons_dir, nom_img))
                
                stats['ok'] += 1
                
            except Exception as e:
                try:
                    Image.fromarray(image_source).save(os.path.join(out_desc_dir, nom_img))
                except:
                    pass
                print(" Error en {}: {}".format(nom_img, e))
                stats['errors'] += 1

