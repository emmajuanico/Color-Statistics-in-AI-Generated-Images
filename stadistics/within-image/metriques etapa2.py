import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color as skcolor
from scipy import stats as scipy_stats

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
COLORS = ["black", "blue", "gray", "green", "red", "white", "yellow"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


#srgb -> hcl
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
    C = np.sqrt(a**2 + b**2)
    H = np.degrees(np.arctan2(b, a)) % 360
    C = np.clip(C, 0, 100)
    return H, C, L


def mean_circular(angles_deg):
    r = np.radians(angles_deg)
    return float(
        np.degrees(np.arctan2(
            np.mean(np.sin(r)),
            np.mean(np.cos(r))
        )) % 360
    )

def std_circular(angles_deg):
    r = np.radians(angles_deg)
    R = np.sqrt(np.mean(np.sin(r))**2 +
                np.mean(np.cos(r))**2)
    R = np.clip(R, 1e-10, 1.0)
    return float(np.degrees(np.sqrt(-2 * np.log(R))))


def calcular_descriptius_H(arr):
    if len(arr) == 0:
        return {k: None for k in [
            'n','mean','median','sd','kurtosi','skewness',
            'p5','p25','p75','p95']}
    return {
        'n':        int(len(arr)),
        'mean':     round(mean_circular(arr),              4),  # circular
        'median':   round(float(np.median(arr)),           4),  # aritmètica
        'sd':       round(std_circular(arr),               4),  # circular
        'kurtosi':  round(float(scipy_stats.kurtosis(arr, fisher=True)), 4),
        'skewness': round(float(scipy_stats.skew(arr)),    4),
        'p5':       round(float(np.percentile(arr,  5)),   4),
        'p25':      round(float(np.percentile(arr, 25)),   4),
        'p75':      round(float(np.percentile(arr, 75)),   4),
        'p95':      round(float(np.percentile(arr, 95)),   4),
    }


def calcular_descriptius(arr):
    if len(arr) == 0:
        return {k: None for k in [
            'n','mean','median','sd','kurtosi','skewness',
            'p5','p25','p75','p95']}
    return {
        'n':        int(len(arr)),
        'mean':     round(float(np.mean(arr)),   4),
        'median':   round(float(np.median(arr)), 4),
        'sd':       round(float(np.std(arr)),    4),
        'kurtosi':  round(float(scipy_stats.kurtosis(arr, fisher=True)), 4),
        'skewness': round(float(scipy_stats.skew(arr)),  4),
        'p5':       round(float(np.percentile(arr,  5)), 4),
        'p25':      round(float(np.percentile(arr, 25)), 4),
        'p75':      round(float(np.percentile(arr, 75)), 4),
        'p95':      round(float(np.percentile(arr, 95)), 4),
    }


rows = []

for model_name, base_path in MODELS.items():
    print("\n=== {} ===".format(model_name.upper()))

    for col in COLORS:
        print("  Color: {}".format(col))

        acum = {
            'objecte': {'H': [], 'C': [], 'L': []},
            'fons':    {'H': [], 'C': [], 'L': []},
        }
        n_imatges = {'objecte': 0, 'fons': 0}

        for obj in TARGET_OBJECTS:
            for segment in ['objecte', 'fons']:
                seg_dir = os.path.join(base_path, obj, col, segment)
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
                        acum[segment]['H'].append(H)
                        acum[segment]['C'].append(C)
                        acum[segment]['L'].append(L)
                        n_imatges[segment] += 1
                    except Exception as e:
                        print("    [ERROR] {}/{}: {}".format(
                            col, nom_img, e))

        for segment in ['objecte', 'fons']:
            if len(acum[segment]['H']) == 0:
                continue

            H_all = np.concatenate(acum[segment]['H'])
            C_all = np.concatenate(acum[segment]['C'])
            L_all = np.concatenate(acum[segment]['L'])

            print("    {} {}: {} imatges, {:,} píxels".format(
                col, segment, n_imatges[segment], len(H_all)))

            desc_H = calcular_descriptius_H(H_all)
            desc_C = calcular_descriptius(C_all)
            desc_L = calcular_descriptius(L_all)

            row = {
                'model':     model_name,
                'color':     col,
                'segment':   segment,
                'n_imatges': n_imatges[segment],
                'n_pixels':  int(len(H_all)),
            }
            for k, v in desc_H.items():
                row['H_{}'.format(k)] = v
            for k, v in desc_C.items():
                row['C_{}'.format(k)] = v
            for k, v in desc_L.items():
                row['L_{}'.format(k)] = v

            rows.append(row)


df = pd.DataFrame(rows)

cols_base = ['model', 'color', 'segment', 'n_imatges', 'n_pixels']
cols_H = ['H_n','H_mean','H_median','H_sd','H_kurtosi',
          'H_skewness','H_p5','H_p25','H_p75','H_p95']
cols_C = ['C_n','C_mean','C_median','C_sd','C_kurtosi',
          'C_skewness','C_p5','C_p25','C_p75','C_p95']
cols_L = ['L_n','L_mean','L_median','L_sd','L_kurtosi',
          'L_skewness','L_p5','L_p25','L_p75','L_p95']

df = df[cols_base + cols_H + cols_C + cols_L]
output_path = os.path.join(OUTPUT_DIR, "within_image_stats_correcte.csv")
df.to_csv(output_path, index=False)

# table delta-chroma
rows_delta = []
for model_name in MODELS.keys():
    for col in COLORS:
        obj_row  = df[(df['model']==model_name) &
                      (df['color']==col) &
                      (df['segment']=='objecte')]
        fons_row = df[(df['model']==model_name) &
                      (df['color']==col) &
                      (df['segment']=='fons')]
        if len(obj_row) > 0 and len(fons_row) > 0:
            med_obj  = obj_row['C_median'].values[0]
            med_fons = fons_row['C_median'].values[0]
            rows_delta.append({
                'model':          model_name,
                'color':          col,
                'C_median_obj':   med_obj,
                'C_median_fons':  med_fons,
                'C_median_delta': round(med_obj - med_fons, 4),
            })

df_delta = pd.DataFrame(rows_delta)
delta_path = os.path.join(OUTPUT_DIR, "within_image_chroma_delta_correcte.csv")
df_delta.to_csv(delta_path, index=False)

print("\n" + "="*60)
print("RESULTAT")
print("="*60)
print("Files CSV principal:   {}".format(len(df)))
print("Files CSV delta:       {}".format(len(df_delta)))
print("Fitxers guardats:")
print("  {}".format(output_path))
print("  {}".format(delta_path))