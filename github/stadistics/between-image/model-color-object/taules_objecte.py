import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ==============================
# CONFIGURACIÓ
# ==============================
CSV_PATH = "" #ourput metriques_etapa1.py
OUTPUT_DIR = ""

MODELS   = ["sdxl_base", "sdxl_turbo", "sd3", "real"]
OBJECTS  = ["bucket", "pen", "coffee mug", "chair"]
COLORS_CROMATIC  = ["red", "green", "blue", "yellow"]
COLORS_ACROMATIC = ["black", "white", "gray"]
COLORS_ALL = ["black", "blue", "gray", "green", "red", "white", "yellow"]

MODEL_NAMES = {
    "sdxl_base":  "SDXL Base",
    "sdxl_turbo": "SDXL Turbo",
    "sd3":        "SD3",
    "real":       "Real",
}
OBJECT_NAMES = {
    "bucket":     "Bucket",
    "pen":        "Pen",
    "coffee mug": "Coffee Mug",
    "chair":      "Chair",
}


df = pd.read_csv(CSV_PATH)

def get_group(df, model, color, obj):
    mask = (df["model"] == model) & (df["color"] == color) & (df["objecte"] == obj)
    return df[mask]



def mean_circular(series):
    vals = series.dropna().values
    if len(vals) == 0:
        return np.nan
    r = np.radians(vals)
    return float(
        np.degrees(np.arctan2(
            np.mean(np.sin(r)),
            np.mean(np.cos(r))
        )) % 360
    )

def std_circular(series):
    vals = series.dropna().values
    if len(vals) < 2:
        return np.nan
    r = np.radians(vals)
    R = np.sqrt(np.mean(np.sin(r))**2 +
                np.mean(np.cos(r))**2)
    R = np.clip(R, 1e-10, 1.0)
    return float(np.degrees(np.sqrt(-2 * np.log(R))))


# taula 1: hue 
rows_ta = []
for obj in OBJECTS:
    for color in COLORS_ALL:
        row_M  = {"Objecte": OBJECT_NAMES[obj], "Color": color.capitalize(), "Stat": "M"}
        row_SD = {"Objecte": OBJECT_NAMES[obj], "Color": color.capitalize(), "Stat": "SD"}
        for model in MODELS:
            for tipus, label in [("objecte", "Obj"), ("fons", "Bg")]:
                col = "{}_mean_H".format(tipus)
                col_key = "{} {}".format(MODEL_NAMES[model], label)
                grp = get_group(df, model, color, obj).dropna(subset=[col])
                if len(grp) > 0:
                    row_M[col_key]  = round(mean_circular(grp[col]), 2)  # circular
                    row_SD[col_key] = round(std_circular(grp[col]),  2)  # circular
                else:
                    row_M[col_key]  = "-"
                    row_SD[col_key] = "-"
        rows_ta.append(row_M)
        rows_ta.append(row_SD)

df_ta = pd.DataFrame(rows_ta)
df_ta.to_csv("{}/taula_A_hue_per_objecte.csv".format(OUTPUT_DIR), index=False)
print(df_ta.to_string(index=False))
print()


# taula 2: chroma 

rows_tb = []
for obj in OBJECTS:
    for color in COLORS_ALL:
        row_M  = {"Objecte": OBJECT_NAMES[obj], "Color": color.capitalize(), "Stat": "M"}
        row_SD = {"Objecte": OBJECT_NAMES[obj], "Color": color.capitalize(), "Stat": "SD"}
        for model in MODELS:
            for tipus, label in [("objecte", "Obj"), ("fons", "Bg")]:
                col = "{}_mean_C".format(tipus)
                col_key = "{} {}".format(MODEL_NAMES[model], label)
                grp = get_group(df, model, color, obj).dropna(subset=[col])
                if len(grp) > 0:
                    row_M[col_key]  = round(grp[col].mean(), 2)
                    row_SD[col_key] = round(grp[col].std(),  2)
                else:
                    row_M[col_key]  = "-"
                    row_SD[col_key] = "-"
        rows_tb.append(row_M)
        rows_tb.append(row_SD)

df_tb = pd.DataFrame(rows_tb)
df_tb.to_csv("{}/taula_B_chroma_per_objecte.csv".format(OUTPUT_DIR), index=False)
print(df_tb.to_string(index=False))
print()


# taula 3: lightness 
rows_tc = []
for obj in OBJECTS:
    for color in COLORS_ALL:
        for model in MODELS:
            grp = get_group(df, model, color, obj)
            row = {
                "Objecte": OBJECT_NAMES[obj],
                "Color":   color.capitalize(),
                "Model":   MODEL_NAMES[model],
                "N":       len(grp),
            }
            for tipus in ["objecte", "fons", "original"]:
                col = "{}_mean_L".format(tipus)
                grp_v = grp.dropna(subset=[col])
                if len(grp_v) > 0:
                    row["{} M".format(tipus.capitalize())]  = round(grp_v[col].mean(), 2)
                    row["{} SD".format(tipus.capitalize())] = round(grp_v[col].std(),  2)
                else:
                    row["{} M".format(tipus.capitalize())]  = "-"
                    row["{} SD".format(tipus.capitalize())] = "-"
            rows_tc.append(row)

df_tc = pd.DataFrame(rows_tc)
df_tc.to_csv("{}/taula_C_lightness_per_objecte.csv".format(OUTPUT_DIR), index=False)
print(df_tc.to_string(index=False))
print()


# taula 4: variancies
rows_td = []
for obj in OBJECTS:
    for color in COLORS_ALL:
        for tipus in ["objecte", "fons", "original"]:
            col_H = "{}_mean_H".format(tipus)
            col_C = "{}_mean_C".format(tipus)
            row = {
                "Objecte": OBJECT_NAMES[obj],
                "Color":   color.capitalize(),
                "Tipus":   tipus.capitalize(),
            }
            for model in MODELS:
                grp = get_group(df, model, color, obj).dropna(subset=[col_H, col_C])
                if len(grp) >= 3:
                    H_rad = np.radians(grp[col_H].values)
                    C_vals = grp[col_C].values
                    X = C_vals * np.cos(H_rad)
                    Y = C_vals * np.sin(H_rad)
                    pca = PCA(n_components=2)
                    pca.fit(np.column_stack([X, Y]))
                    row[MODEL_NAMES[model]] = round(float(np.sum(pca.explained_variance_)), 4)
                else:
                    row[MODEL_NAMES[model]] = "-"
            rows_td.append(row)

df_td = pd.DataFrame(rows_td)
df_td.to_csv("{}/taula_D_bivariate_per_objecte.csv".format(OUTPUT_DIR), index=False)
print(df_td.to_string(index=False))
print()


print("=" * 60)
print("FITXERS GENERATS (Hue amb estadístiques circulars):")
print("  taula_A_hue_per_objecte.csv")
print("  taula_B_chroma_per_objecte.csv")
print("  taula_C_lightness_per_objecte.csv")
print("  taula_D_bivariate_per_objecte.csv")
print("=" * 60)