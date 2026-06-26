import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color as skcolor
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

BASE_IA    = "" #segmented ia images
BASE_REALS = "" #segmented real images
OUTPUT_DIR = ""

MODELS = {
    "sdxl_base":  os.path.join(BASE_IA, "sdxl_base"),
    "sdxl_turbo": os.path.join(BASE_IA, "sdxl_turbo"),
    "sd3":        os.path.join(BASE_IA, "sd3"),
    "real":       BASE_REALS,
}
TARGET_OBJECTS = ["bucket", "pen", "coffee mug", "chair"]
COLORS_ALL     = ["black", "blue", "gray", "green", "red", "white", "yellow"]
SEGMENTS       = ["objecte", "fons"]

COMPARISONS = [
    ("sdxl_base",  "real"),
    ("sdxl_turbo", "real"),
    ("sd3",        "real"),
    ("sdxl_base",  "sdxl_turbo"),
    ("sdxl_base",  "sd3"),
    ("sdxl_turbo", "sd3"),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# rgb -> hcl
def rgb_to_hcl(img_rgba):
    arr = np.array(img_rgba, dtype=np.float32)
    if arr.shape[2] == 4:
        mascara = arr[:, :, 3] > 0
    else:
        mascara = np.ones(arr.shape[:2], dtype=bool)
    if mascara.sum() == 0:
        return None, None, None
    pixels_rgb = arr[:, :, :3][mascara] / 255.0
    pixels_lab = skcolor.rgb2lab(
        pixels_rgb.reshape(1, -1, 3)).reshape(-1, 3)
    L = pixels_lab[:, 0]
    a = pixels_lab[:, 1]
    b = pixels_lab[:, 2]
    C = np.clip(np.sqrt(a**2 + b**2), 0, 100)
    H = np.degrees(np.arctan2(b, a)) % 360
    return H, C, L


def acumular_pixels(base_path, color, segment):
    H_all, C_all, L_all = [], [], []
    for obj in TARGET_OBJECTS:
        seg_dir = os.path.join(base_path, obj, color, segment)
        if not os.path.isdir(seg_dir):
            continue
        for nom_img in sorted(os.listdir(seg_dir)):
            if not nom_img.endswith(".png"):
                continue
            try:
                img = Image.open(
                    os.path.join(seg_dir, nom_img)).convert("RGBA")
                H, C, L = rgb_to_hcl(img)
                if H is None:
                    continue
                H_all.append(H)
                C_all.append(C)
                L_all.append(L)
            except Exception as e:
                print("  [ERROR] {}/{}: {}".format(color, nom_img, e))
    if len(H_all) == 0:
        return None, None, None
    return (np.concatenate(H_all),
            np.concatenate(C_all),
            np.concatenate(L_all))


def fmt_p(p):
    if p < 0.001:
        return "<0.001"
    return round(p, 4)


def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"



cache = {}

def get_pixels(model_name, color, segment):
    key = (model_name, color, segment)
    if key not in cache:
        print("  Carregant {} × {} × {}...".format(
            model_name, color, segment))
        H, C, L = acumular_pixels(MODELS[model_name], color, segment)
        cache[key] = (H, C, L)
    return cache[key]


# ks tests
rows = []

for color in COLORS_ALL:
    print("\nColor: {}".format(color))
    for segment in SEGMENTS:
        for m1, m2 in COMPARISONS:
            H1, C1, L1 = get_pixels(m1, color, segment)
            H2, C2, L2 = get_pixels(m2, color, segment)

            if H1 is None or H2 is None:
                continue

            MAX_N = 50_000
            np.random.seed(42)
            idx1 = np.random.choice(len(H1), min(MAX_N, len(H1)),
                                    replace=False)
            idx2 = np.random.choice(len(H2), min(MAX_N, len(H2)),
                                    replace=False)

            # KS test per H, C, L
            ks_H = stats.ks_2samp(H1[idx1], H2[idx2])
            ks_C = stats.ks_2samp(C1[idx1], C2[idx2])
            ks_L = stats.ks_2samp(L1[idx1], L2[idx2])

            row = {
                "color":    color,
                "segment":  segment,
                "model_1":  m1,
                "model_2":  m2,
                "n_1":      len(H1),
                "n_2":      len(H2),
                # To (H)
                "H_D":      round(ks_H.statistic, 4),
                "H_p":      fmt_p(ks_H.pvalue),
                "H_sig":    sig_label(ks_H.pvalue),
                # Chroma (C)
                "C_D":      round(ks_C.statistic, 4),
                "C_p":      fmt_p(ks_C.pvalue),
                "C_sig":    sig_label(ks_C.pvalue),
                # Lightness (L)
                "L_D":      round(ks_L.statistic, 4),
                "L_p":      fmt_p(ks_L.pvalue),
                "L_sig":    sig_label(ks_L.pvalue),
            }
            rows.append(row)
            print("  {} vs {} | {} | H_D={:.3f} C_D={:.3f} L_D={:.3f}".format(
                m1, m2, segment,
                ks_H.statistic, ks_C.statistic, ks_L.statistic))


df = pd.DataFrame(rows)
output_path = os.path.join(OUTPUT_DIR, "taula10_ks_test.csv")
df.to_csv(output_path, index=False)