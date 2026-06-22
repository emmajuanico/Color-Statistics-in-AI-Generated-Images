import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

CSV_STATS = "" #output metriques_etapa2.py "within_image_stats_correcte.csv"
CSV_DELTA = "" # output metriques_etapa2.py within_image_chroma_delta_correcte.csv
OUTPUT_DIR = ""

MODELS = ["sdxl_base", "sdxl_turbo", "sd3", "real"]
MODEL_NAMES = {
    "sdxl_base":  "SDXL Base",
    "sdxl_turbo": "SDXL Turbo",
    "sd3":        "SD3",
    "real":       "Real",
}
COLORS_ALL      = ["black", "blue", "gray", "green", "red", "white", "yellow"]
COLORS_CROMATIC = ["blue", "green", "red", "yellow"]
COLOR_NAMES = {
    "black": "Negre", "blue": "Blau",   "gray":   "Gris",
    "green": "Verd",  "red":  "Vermell","white":  "Blanc",
    "yellow":"Groc",
}


df   = pd.read_csv(CSV_STATS)
df_d = pd.read_csv(CSV_DELTA)


def get_row(model, color, segment):
    mask = ((df['model']   == model) &
            (df['color']   == color) &
            (df['segment'] == segment))
    r = df[mask]
    return r.iloc[0] if len(r) > 0 else None


def fmt(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    return round(float(val), decimals)


# taula: to
rows_t6 = []
for color in COLORS_ALL:
    for segment in ["objecte", "fons"]:
        row = {
            "Color":   COLOR_NAMES[color],
            "Segment": segment.capitalize(),
        }
        for model in MODELS:
            r = get_row(model, color, segment)
            p = MODEL_NAMES[model]
            if r is not None:
                row["{} N px".format(p)]    = int(r['H_n']) if not pd.isna(r['H_n']) else "-"
                row["{} Mean".format(p)]    = fmt(r['H_mean'])
                row["{} Median".format(p)]  = fmt(r['H_median'])
                row["{} SD".format(p)]      = fmt(r['H_sd'])
                row["{} Kurt.".format(p)]   = fmt(r['H_kurtosi'], 3)
                row["{} Skew.".format(p)]   = fmt(r['H_skewness'], 3)
            else:
                for s in ["N px","Mean","Median","SD","Kurt.","Skew."]:
                    row["{} {}".format(p, s)] = "-"
        rows_t6.append(row)

df_t6 = pd.DataFrame(rows_t6)
df_t6.to_csv("{}/taula6_within_hue.csv".format(OUTPUT_DIR), index=False)
print(df_t6.to_string(index=False))


# taula: croma
rows_t7 = []
for color in COLORS_ALL:
    for segment in ["objecte", "fons"]:
        row = {
            "Color":   COLOR_NAMES[color],
            "Segment": segment.capitalize(),
        }
        for model in MODELS:
            r = get_row(model, color, segment)
            p = MODEL_NAMES[model]
            if r is not None:
                row["{} Mean".format(p)]   = fmt(r['C_mean'])
                row["{} Median".format(p)] = fmt(r['C_median'])
                row["{} SD".format(p)]     = fmt(r['C_sd'])
                row["{} Kurt.".format(p)]  = fmt(r['C_kurtosi'], 3)
                row["{} Skew.".format(p)]  = fmt(r['C_skewness'], 3)
            else:
                for s in ["Mean","Median","SD","Kurt.","Skew."]:
                    row["{} {}".format(p, s)] = "-"
        rows_t7.append(row)

df_t7 = pd.DataFrame(rows_t7)
df_t7.to_csv("{}/taula7_within_chroma.csv".format(OUTPUT_DIR), index=False)
print(df_t7.to_string(index=False))


# taula: lluminositat
rows_t8 = []
for color in COLORS_ALL:
    for segment in ["objecte", "fons"]:
        row = {
            "Color":   COLOR_NAMES[color],
            "Segment": segment.capitalize(),
        }
        for model in MODELS:
            r = get_row(model, color, segment)
            p = MODEL_NAMES[model]
            if r is not None:
                row["{} Mean".format(p)]   = fmt(r['L_mean'])
                row["{} Median".format(p)] = fmt(r['L_median'])
                row["{} SD".format(p)]     = fmt(r['L_sd'])
                row["{} Kurt.".format(p)]  = fmt(r['L_kurtosi'], 3)
                row["{} Skew.".format(p)]  = fmt(r['L_skewness'], 3)
            else:
                for s in ["Mean","Median","SD","Kurt.","Skew."]:
                    row["{} {}".format(p, s)] = "-"
        rows_t8.append(row)

df_t8 = pd.DataFrame(rows_t8)
df_t8.to_csv("{}/taula8_within_lightness.csv".format(OUTPUT_DIR), index=False)
print(df_t8.to_string(index=False))


# taula: croma diferencies
rows_t9 = []
for color in COLORS_ALL:
    row = {"Color": COLOR_NAMES[color]}
    for model in MODELS:
        mask = ((df_d['model'] == model) & (df_d['color'] == color))
        r = df_d[mask]
        p = MODEL_NAMES[model]
        if len(r) > 0:
            row["{} Obj".format(p)]  = fmt(r['C_median_obj'].values[0])
            row["{} Fons".format(p)] = fmt(r['C_median_fons'].values[0])
            row["{} Δ".format(p)]    = fmt(r['C_median_delta'].values[0])
        else:
            for s in ["Obj","Fons","Δ"]:
                row["{} {}".format(p, s)] = "-"
    rows_t9.append(row)

df_t9 = pd.DataFrame(rows_t9)
df_t9.to_csv("{}/taula9_chroma_delta.csv".format(OUTPUT_DIR), index=False)
print(df_t9.to_string(index=False))


print("\n" + "="*60)
print("FITXERS GENERATS")
print("="*60)
print("  taula6_within_hue.csv")
print("  taula7_within_chroma.csv")
print("  taula8_within_lightness.csv")
print("  taula9_chroma_delta.csv")