import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


CSV_PATH = "" # output metriques_etapa1.py
OUTPUT_DIR = ""

MODELS = ["sdxl_base", "sdxl_turbo", "sd3", "real"]
COLORS_CROMATIC = ["red", "green", "blue", "yellow", "gray", "white", "black"]

MODEL_NAMES = {
    "sdxl_base":  "SDXL Base",
    "sdxl_turbo": "SDXL Turbo",
    "sd3":        "Stable Diffusion 3",
    "real":       "Imatges Reals",
}

COLOR_MAP = {
    "red":    "#C62828",
    "green":  "#2E7D32",
    "blue":   "#1565C0",
    "yellow": "#F57F17",
    "gray" :  "#979797",
    "white":  "#DEDEDE",
    "black" : "#000000"
}

COLOR_LABELS_CAT = {
    "red": "Vermell", "green": "Verd", "blue": "Blau", 
    "yellow": "Groc", "gray": "Gris", "white": "Blanc", "black": "Negre"
}

Z_LAYERS = {
    "original": 0.00,
    "objecte":  0.80,
    "fons":     1.60,
}

LAYER_LABELS = {
    "original": "Imatges Originals",
    "objecte":  "Objectes",
    "fons":     "Fons",
}

try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    df = pd.DataFrame(columns=['model', 'color']) 

#functions
def draw_reference_plane(ax, z, r_max=105):
    """
    Pla de referència amb cercle cromàtic colorejat al pla base,
    cercles concèntrics i línies radials.
    """
    theta = np.linspace(0, 2*np.pi, 721)

    if z == 0:
        for i in range(len(theta)-1):
            hue_deg = np.degrees(theta[i]) % 360
            h = hue_deg / 60.0
            xi = int(h) % 6
            f = h - int(h)
            rgb_table = [
                (1, f, 0), (1-f, 1, 0), (0, 1, f),
                (0, 1-f, 1), (f, 0, 1), (1, 0, 1-f)
            ]
            r_c, g_c, b_c = rgb_table[xi]
            ax.plot(
                [r_max*np.cos(theta[i]),   r_max*np.cos(theta[i+1])],
                [r_max*np.sin(theta[i]),   r_max*np.sin(theta[i+1])],
                [z, z],
                color=(r_c, g_c, b_c), linewidth=5, alpha=0.55,
                solid_capstyle='butt', zorder=2
            )
        for r in np.linspace(85, r_max, 6):
            for i in range(len(theta)-1):
                hue_deg = np.degrees(theta[i]) % 360
                h = hue_deg / 60.0
                xi = int(h) % 6
                f = h - int(h)
                rgb_table = [
                    (1, f, 0), (1-f, 1, 0), (0, 1, f),
                    (0, 1-f, 1), (f, 0, 1), (1, 0, 1-f)
                ]
                r_c, g_c, b_c = rgb_table[xi]
                alpha = 0.18 * (r / r_max)
                ax.plot(
                    [r*np.cos(theta[i]),  r*np.cos(theta[i+1])],
                    [r*np.sin(theta[i]),  r*np.sin(theta[i+1])],
                    [z, z],
                    color=(r_c, g_c, b_c), linewidth=2.5,
                    alpha=alpha, solid_capstyle='butt', zorder=1
                )

    theta_c = np.linspace(0, 2*np.pi, 361)
    for r in [25, 50, 75, 100]:
        lw = 0.8 if r == 100 else 0.4
        ax.plot(r*np.cos(theta_c), r*np.sin(theta_c), np.full(361, z),
                color='#999999', linewidth=lw, alpha=0.5, zorder=3)

    for angle_deg in range(0, 360, 45):
        ar = np.radians(angle_deg)
        ax.plot([0, r_max*np.cos(ar)], [0, r_max*np.sin(ar)], [z, z],
                color='#BBBBBB', linewidth=0.4, alpha=0.5, zorder=3)

    if z == 0:
        for angle_deg, label in [(0,'0°'),(90,'90°'),(180,'180°'),(270,'270°')]:
            ar = np.radians(angle_deg)
            ax.text((r_max+18)*np.cos(ar), (r_max+18)*np.sin(ar), z,
                    label, fontsize=7.5, color='#666666',
                    ha='center', va='center', style='italic',
                    fontfamily='serif', zorder=4)
        # Etiquetes de croma
        for r, lbl in [(50,'50'), (100,'100')]:
            ax.text(r*np.cos(np.radians(40))+2,
                    r*np.sin(np.radians(40))+2, z,
                    lbl, fontsize=6, color='#999999',
                    ha='left', va='bottom', zorder=4)


def pca_ellipse_3d(ax, X, Y, z_base, color): #elipses confiança pca
    if len(X) < 4:
        return
    data = np.column_stack([X, Y])
    pca  = PCA(n_components=2)
    pca.fit(data)

    chi2_95    = 5.991
    evals      = pca.explained_variance_
    width      = 2 * np.sqrt(chi2_95 * evals[0])
    height     = 2 * np.sqrt(chi2_95 * evals[1])
    angle_rad  = np.arctan2(*pca.components_[0][::-1])
    cx, cy     = pca.mean_

    t  = np.linspace(0, 2*np.pi, 300)
    ex = (width/2)  * np.cos(t)
    ey = (height/2) * np.sin(t)
    rx = cx + ex*np.cos(angle_rad) - ey*np.sin(angle_rad)
    ry = cy + ex*np.sin(angle_rad) + ey*np.cos(angle_rad)
    rz = np.full(300, z_base)

    ax.plot(rx, ry, rz,
            color=color, linewidth=1.6, alpha=0.95,
            zorder=10, solid_capstyle='round')


def generate_figure(model_name, df, output_dir):
    fig = plt.figure(figsize=(11, 9))  
    fig.patch.set_facecolor('white')
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = True
        pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        pane.set_edgecolor('#E0E0E0')

    ax.grid(True, color='#F2F2F2', linewidth=0.4, linestyle='-')

    for layer, z_val in Z_LAYERS.items():
        draw_reference_plane(ax, z=z_val)

    ax.plot([0,0],[0,0],
            [Z_LAYERS["original"]-0.05, Z_LAYERS["fons"]+0.1],
            color='#CCCCCC', linewidth=0.8, alpha=0.7)

    for layer, z_base in Z_LAYERS.items():
        col_H = "{}_mean_H".format(layer)
        col_C = "{}_mean_C".format(layer)

        for color in COLORS_CROMATIC:
            mask = (df['model'] == model_name) & (df['color'] == color)
            grp  = df[mask].dropna(subset=[col_H, col_C])
            if len(grp) == 0:
                continue

            H_rad  = np.radians(grp[col_H].values)
            C_vals = grp[col_C].values
            X = C_vals * np.cos(H_rad)
            Y = C_vals * np.sin(H_rad)
            Z = np.full(len(X), z_base)
            c = COLOR_MAP[color]

            ax.scatter(X, Y, Z, c=c, s=5, alpha=0.50,
                       edgecolors='none', zorder=6, depthshade=False)
            pca_ellipse_3d(ax, X, Y, z_base, c)
    for layer, z_val in Z_LAYERS.items():
        ax.text(-135, 0, z_val, LAYER_LABELS[layer] + " —",
                fontsize=9, color='#222222',
                ha='right', va='center',
                fontfamily='serif', style='italic', zorder=20)
    #llegenda
    handles = [
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor=COLOR_MAP[c], markersize=5,
               markeredgewidth=0, label=COLOR_LABELS_CAT[c])
        for c in COLORS_CROMATIC
    ]
    handles += [
        Line2D([0],[0], color='#555555', linewidth=1.2,
               label='El·lipse IC 95%')
    ]
    
    leg = ax.legend(
        handles=handles,
        loc='upper right',
        fontsize=7.5,
        framealpha=0.95,
        edgecolor='#CCCCCC',
        borderpad=0.5,
        handlelength=1.4,
        labelspacing=0.3,
        bbox_to_anchor=(1.15, 0.95),
        title='Indicador de color',
        title_fontsize=8,
    )
    leg.get_frame().set_linewidth(0.5)

    # titol
    ax.set_title(
        "Distribucions de Tonalitat/Croma — {}".format(MODEL_NAMES[model_name]),
        fontsize=11, pad=25, color='#111111',
        fontfamily='serif', loc='left', x=-0.05
    )

    # eixos
    lim = 125
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-0.1, Z_LAYERS["fons"] + 0.3)

    ax.set_xlabel("a*", fontsize=9, labelpad=6, color='#444444', fontfamily='serif')
    ax.set_ylabel("b*", fontsize=9, labelpad=6, color='#444444', fontfamily='serif')
    ax.set_zlabel("Altura de Capa (Z)", fontsize=9, labelpad=10, color='#444444', fontfamily='serif')

    ax.set_zticks(list(Z_LAYERS.values()))
    ax.set_zticklabels(
        ["{:.2f}".format(v) for v in Z_LAYERS.values()],
        fontsize=7, color='#777777')
    ax.tick_params(axis='x', labelsize=7, colors='#888888', pad=2)
    ax.tick_params(axis='y', labelsize=7, colors='#888888', pad=2)

    ax.view_init(elev=20, azim=-55)

    plt.subplots_adjust(left=0.05, right=0.82, top=0.90, bottom=0.05)

    fname = "{}/figura1_{}.png".format(output_dir, model_name)
    plt.savefig(fname, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    print("  Guardada: {}".format(fname))
    plt.close()


if len(df) > 0:
    for model in MODELS:
        generate_figure(model, df, OUTPUT_DIR)
else:
    print("\nNo s'ha pogut processar el bucle perquè el fitxer de dades està buit o no s'ha trobat.")