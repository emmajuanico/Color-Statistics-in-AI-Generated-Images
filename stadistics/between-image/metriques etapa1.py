import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color as skcolor

BASE_IA    = "" #ia segmented images
BASE_REALS = "" #real segmented images
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

pixels_total     = 0
pixels_rescalats = 0
exclusions_buits = 0

#srgb -> hcl
def rgb_to_hcl(img_rgba):
    global pixels_total, pixels_rescalats, exclusions_buits

    arr = np.array(img_rgba, dtype=np.float32)
    if arr.shape[2] == 4:
        mascara = arr[:, :, 3] > 0
    else:
        mascara = np.ones(arr.shape[:2], dtype=bool)

    if mascara.sum() == 0:
        exclusions_buits += 1
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

    n = len(C)
    pixels_total     += n
    pixels_rescalats += int(np.sum(C >= 100))

    return H, C, L


#mitjana circular per al to
def mean_circular(H):
    r = np.radians(H)
    return float(
        np.degrees(np.arctan2(np.mean(np.sin(r)),
                              np.mean(np.cos(r)))) % 360
    )


def calcular_stats(H, C, L):

    return {
        "mean_H":   round(mean_circular(H), 4), 
        "mean_C":   round(float(np.mean(C)), 4),
        "mean_L":   round(float(np.mean(L)), 4),
        "n_pixels": int(len(H)),
    }


#bucle
rows   = []
total  = 0
errors = 0

for model_name, base_path in MODELS.items():

    for obj in TARGET_OBJECTS:
        obj_dir = os.path.join(base_path, obj)
        if not os.path.isdir(obj_dir):
            continue

        for col in COLORS:
            col_dir   = os.path.join(obj_dir, col)
            obj_dir2  = os.path.join(col_dir, "objecte")
            fons_dir2 = os.path.join(col_dir, "fons")

            if not os.path.isdir(obj_dir2) or not os.path.isdir(fons_dir2):
                continue

            imatges = sorted([f for f in os.listdir(obj_dir2)
                              if f.endswith(".png")])

            for nom_img in imatges:
                path_obj  = os.path.join(obj_dir2, nom_img)
                path_fons = os.path.join(fons_dir2, nom_img)

                if not os.path.exists(path_fons):
                    continue

                row = {
                    "model":    model_name,
                    "objecte":  obj,
                    "color":    col,
                    "filename": nom_img,
                }
                valida = True

                for tipus, path in [("objecte", path_obj),
                                    ("fons",    path_fons)]:
                    try:
                        img = Image.open(path).convert("RGBA")
                        H, C, L = rgb_to_hcl(img)
                        if H is None:
                            valida = False
                            break
                        stats = calcular_stats(H, C, L)
                        row["{}_mean_H".format(tipus)]   = stats["mean_H"]
                        row["{}_mean_C".format(tipus)]   = stats["mean_C"]
                        row["{}_mean_L".format(tipus)]   = stats["mean_L"]
                        row["{}_n_pixels".format(tipus)] = stats["n_pixels"]
                    except Exception as e:
                        print(" error {}/{}: {}".format(
                            tipus, nom_img, e))
                        errors += 1
                        valida = False
                        break

                if valida:
                    try:
                        img_obj  = Image.open(path_obj).convert("RGBA")
                        img_fons = Image.open(path_fons).convert("RGBA")
                        arr_obj  = np.array(img_obj,  dtype=np.float32)
                        arr_fons = np.array(img_fons, dtype=np.float32)

                        mascara_obj  = arr_obj[:, :, 3] > 0
                        mascara_fons = arr_fons[:, :, 3] > 0

                        arr_combined = arr_obj.copy()
                        arr_combined[mascara_fons & ~mascara_obj] = \
                            arr_fons[mascara_fons & ~mascara_obj]

                        img_combined = Image.fromarray(
                            arr_combined.astype(np.uint8))
                        H_o, C_o, L_o = rgb_to_hcl(img_combined)

                        if H_o is not None:
                            stats_o = calcular_stats(H_o, C_o, L_o)
                            row["original_mean_H"]   = stats_o["mean_H"]
                            row["original_mean_C"]   = stats_o["mean_C"]
                            row["original_mean_L"]   = stats_o["mean_L"]
                            row["original_n_pixels"] = stats_o["n_pixels"]
                        else:
                            row["original_mean_H"]   = None
                            row["original_mean_C"]   = None
                            row["original_mean_L"]   = None
                            row["original_n_pixels"] = 0

                    except Exception as e:
                        print(" error original/{}: {}".format(
                            nom_img, e))
                        row["original_mean_H"]   = None
                        row["original_mean_C"]   = None
                        row["original_mean_L"]   = None
                        row["original_n_pixels"] = 0

                    rows.append(row)
                    total += 1

#guardar csv
columns = [
    "model", "objecte", "color", "filename",
    "objecte_mean_H", "objecte_mean_C", "objecte_mean_L", "objecte_n_pixels",
    "fons_mean_H",    "fons_mean_C",    "fons_mean_L",    "fons_n_pixels",
    "original_mean_H","original_mean_C","original_mean_L","original_n_pixels",
]

df = pd.DataFrame(rows, columns=columns)
output_path = os.path.join(OUTPUT_DIR, "between_image_stats_circular.csv")
df.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("RESULTAT")
print("=" * 60)
print("Imatges processades:        {}".format(total))
print("Errors:                     {}".format(errors))
print("Excloses (màscara buida):   {}".format(exclusions_buits))
print("Total píxels:               {}".format(pixels_total))
print("píxels rescalats (C≥100):   {} ({:.4f}%)".format(
    pixels_rescalats,
    100 * pixels_rescalats / pixels_total if pixels_total > 0 else 0))
print("Fitxer guardat:             {}".format(output_path))