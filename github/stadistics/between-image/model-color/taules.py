import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

CSV_PATH = r"between_image_stats_circular.csv" #output metriques_etapa1
OUTPUT_DIR = ""

MODELS = ["sdxl_base", "sdxl_turbo", "sd3", "real"]
COLORS_CROMATIC  = ["red", "green", "blue", "yellow"]
COLORS_ACROMATIC = ["black", "white", "gray"]
COLORS_ALL = ["black", "blue", "gray", "green", "red", "white", "yellow"]

MODEL_NAMES = {
    "sdxl_base":  "SDXL Base",
    "sdxl_turbo": "SDXL Turbo",
    "sd3":        "SD3",
    "real":       "Real",
}

#load csv
df = pd.read_csv(CSV_PATH)

def get_group(df, model, color):
    mask = (df["model"] == model) & (df["color"] == color)
    return df[mask]


def mean_circular(series): #circular mean
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

def std_circular(series): #circular sd
    vals = series.dropna().values
    if len(vals) < 2:
        return np.nan
    r = np.radians(vals)
    R = np.sqrt(np.mean(np.sin(r))**2 +
                np.mean(np.cos(r))**2)
    R = np.clip(R, 1e-10, 1.0)
    return float(np.degrees(np.sqrt(-2 * np.log(R))))


# table 1: hue 

rows_t1 = []
for color in COLORS_ALL:
    row_M  = {"Color": color.capitalize(), "Stat": "M"}
    row_SD = {"Color": color.capitalize(), "Stat": "SD"}
    for model in MODELS:
        for tipus, label in [("objecte", "Obj"), ("fons", "Bg")]:
            col = "{}_mean_H".format(tipus)
            col_key = "{} {}".format(MODEL_NAMES[model], label)
            grp = get_group(df, model, color).dropna(subset=[col])
            if len(grp) > 0:
                row_M[col_key]  = round(mean_circular(grp[col]), 2)
                row_SD[col_key] = round(std_circular(grp[col]),  2)
            else:
                row_M[col_key]  = "-"
                row_SD[col_key] = "-"
    rows_t1.append(row_M)
    rows_t1.append(row_SD)

df_t1 = pd.DataFrame(rows_t1)
df_t1.to_csv("{}/taula1_hue_means_sd.csv".format(OUTPUT_DIR), index=False)
print(df_t1.to_string(index=False))
print()


# table 2 Chroma Means i SD

rows_t2 = []
for color in COLORS_ALL:
    row_M  = {"Color": color.capitalize(), "Stat": "M"}
    row_SD = {"Color": color.capitalize(), "Stat": "SD"}
    for model in MODELS:
        for tipus, label in [("objecte", "Obj"), ("fons", "Bg")]:
            col = "{}_mean_C".format(tipus)
            col_key = "{} {}".format(MODEL_NAMES[model], label)
            grp = get_group(df, model, color).dropna(subset=[col])
            if len(grp) > 0:
                row_M[col_key]  = round(grp[col].mean(), 2)
                row_SD[col_key] = round(grp[col].std(),  2)
            else:
                row_M[col_key]  = "-"
                row_SD[col_key] = "-"
    rows_t2.append(row_M)
    rows_t2.append(row_SD)

df_t2 = pd.DataFrame(rows_t2)
df_t2.to_csv("{}/taula2_chroma_means_sd.csv".format(OUTPUT_DIR), index=False)
print(df_t2.to_string(index=False))
print()


# table 3: lightness

rows_t3 = []
for color in COLORS_ALL:
    for model in MODELS:
        grp = get_group(df, model, color)
        row = {
            "Color": color.capitalize(),
            "Model": MODEL_NAMES[model],
            "N":     len(grp),
        }
        for tipus in ["objecte", "fons", "original"]:
            col = "{}_mean_L".format(tipus)
            grp_v = grp.dropna(subset=[col])
            if len(grp_v) > 0:
                row["{} Mean L".format(tipus.capitalize())]   = round(grp_v[col].mean(), 2)
                row["{} SD L".format(tipus.capitalize())]     = round(grp_v[col].std(),  2)
                row["{} Median L".format(tipus.capitalize())] = round(grp_v[col].median(), 2)
            else:
                row["{} Mean L".format(tipus.capitalize())]   = "-"
                row["{} SD L".format(tipus.capitalize())]     = "-"
                row["{} Median L".format(tipus.capitalize())] = "-"
        rows_t3.append(row)

df_t3 = pd.DataFrame(rows_t3)
df_t3.to_csv("{}/taula3_lightness_descriptive.csv".format(OUTPUT_DIR), index=False)
print(df_t3.to_string(index=False))
print()


# table 4 Hue/Chroma total Variances (PCA)

rows_t4 = []
for color in COLORS_ALL:
    for tipus in ["objecte", "fons", "original"]:
        col_H = "{}_mean_H".format(tipus)
        col_C = "{}_mean_C".format(tipus)
        row = {
            "Color": color.capitalize(),
            "Tipus": tipus.capitalize(),
        }
        for model in MODELS:
            grp = get_group(df, model, color).dropna(subset=[col_H, col_C])
            if len(grp) >= 3:
                H_rad = np.radians(grp[col_H].values)
                C_vals = grp[col_C].values
                X = C_vals * np.cos(H_rad)
                Y = C_vals * np.sin(H_rad)
                data = np.column_stack([X, Y])
                pca = PCA(n_components=2)
                pca.fit(data)
                var_total = round(float(np.sum(pca.explained_variance_)), 4)
                row[MODEL_NAMES[model]] = var_total
            else:
                row[MODEL_NAMES[model]] = "-"
        rows_t4.append(row)

df_t4 = pd.DataFrame(rows_t4)
df_t4.to_csv("{}/taula4_bivariate_variances.csv".format(OUTPUT_DIR), index=False)
print(df_t4.to_string(index=False))
print()


#table 5: pearson correlations

rows_t5 = []
for color in COLORS_ALL:
    for model in MODELS:
        grp = get_group(df, model, color).dropna(
            subset=["objecte_mean_H", "fons_mean_H",
                    "objecte_mean_C", "fons_mean_C",
                    "objecte_mean_L", "fons_mean_L"])

        row = {
            "Color": color.capitalize(),
            "Model": MODEL_NAMES[model],
            "N":     len(grp),
        }

        for var, label in [("H", "Hue"), ("C", "Chroma"), ("L", "Lightness")]:
            col_obj  = "objecte_mean_{}".format(var)
            col_fons = "fons_mean_{}".format(var)

            if len(grp) >= 3:
                r, p = stats.pearsonr(grp[col_obj].values, grp[col_fons].values)
                row["r {}".format(label)]  = round(r, 3)
                row["p {}".format(label)]  = round(p, 4)
                if p < 0.001:
                    row["sig {}".format(label)] = "***"
                elif p < 0.01:
                    row["sig {}".format(label)] = "**"
                elif p < 0.05:
                    row["sig {}".format(label)] = "*"
                else:
                    row["sig {}".format(label)] = "ns"
            else:
                row["r {}".format(label)]   = "-"
                row["p {}".format(label)]   = "-"
                row["sig {}".format(label)] = "-"

        rows_t5.append(row)

df_t5 = pd.DataFrame(rows_t5)
df_t5.to_csv("{}/taula5_correlacions_pearson.csv".format(OUTPUT_DIR), index=False)
print(df_t5.to_string(index=False))
print()