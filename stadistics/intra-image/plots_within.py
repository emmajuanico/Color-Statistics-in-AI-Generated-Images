import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from matplotlib import rcParams
from PIL import Image
from skimage import color as skcolor
import warnings
warnings.filterwarnings("ignore")

BASE_IA    = "" #ia segmented images
BASE_REALS = "" #real segmented images
OUTPUT_DIR = ""

MODELS = {
    "sdxl_base":  os.path.join(BASE_IA, "sdxl_base"),
    "sdxl_turbo": os.path.join(BASE_IA, "sdxl_turbo"),
    "sd3":        os.path.join(BASE_IA, "sd3"),
    "real":       BASE_REALS,
}
TARGET_OBJECTS  = ["bucket", "pen", "coffee mug", "chair"]
COLORS_ALL      = ["black", "blue", "gray", "green", "red", "white", "yellow"]
COLORS_CROMATIC = ["blue", "green", "red", "yellow"]

MODEL_LABELS = {
    "sdxl_base":  "(A) SDXL Base",
    "sdxl_turbo": "(B) SDXL Turbo",
    "sd3":        "(C) SD3",
    "real":       "(D) Imatges Reals",
}
COLOR_LINE = {
    "black":  ("#111111", "Negre"),
    "blue":   ("#1565C0", "Blau"),
    "gray":   ("#757575", "Gris"),
    "green":  ("#2E7D32", "Verd"),
    "red":    ("#C62828", "Vermell"),
    "white":  ("#9E9E9E", "Blanc"),
    "yellow": ("#F9A825", "Groc"),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

rcParams['font.family']        = 'serif'
rcParams['font.serif']         = ['Georgia', 'Times New Roman', 'DejaVu Serif']
rcParams['axes.spines.top']    = False
rcParams['axes.spines.right']  = False
rcParams['axes.linewidth']     = 0.8
rcParams['xtick.major.width']  = 0.8
rcParams['ytick.major.width']  = 0.8
rcParams['xtick.labelsize']    = 8
rcParams['ytick.labelsize']    = 8


#srgb ->hcl
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


_cache = {}

def get_pixels(base_path, color, segment):
    key = (base_path, color, segment)
    if key not in _cache:
        print("  Carregant {} × {}...".format(color, segment))
        _cache[key] = acumular_pixels(base_path, color, segment)
    return _cache[key]


def setup_fig():
    fig = plt.figure(figsize=(11, 9))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.38, wspace=0.30,
                           left=0.08, right=0.97,
                           top=0.91, bottom=0.08)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    return fig, axes

def style_ax(ax, model_name):
    ax.set_facecolor('white')
    ax.set_title(MODEL_LABELS[model_name], fontsize=9.5,
                 fontweight='bold', color='#111111',
                 loc='left', pad=6, fontfamily='serif')
    ax.spines['left'].set_color('#BBBBBB')
    ax.spines['bottom'].set_color('#BBBBBB')
    ax.tick_params(colors='#555555', length=3)

def add_legend(ax, handles, loc='upper right'):
    h_solid = mlines.Line2D([], [], color='#555555', linewidth=1.2,
                            linestyle='-', label='Objecte')
    h_dot   = mlines.Line2D([], [], color='#555555', linewidth=1.2,
                            linestyle=':', label='Fons')
    ax.legend(handles=handles + [h_solid, h_dot],
              fontsize=7, framealpha=0.92, edgecolor='#DDDDDD',
              loc=loc, handlelength=2.0,
              ncol=2 if len(handles) > 4 else 1)

def save_fig(fig, fname, suptitle):
    fig.suptitle(suptitle, fontsize=11, fontweight='bold',
                 color='#111111', fontfamily='serif', y=0.97)
    plt.savefig(os.path.join(OUTPUT_DIR, fname),
                dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("  Guardada: {}".format(fname))
    plt.close()


# cdf
def make_cdf(metric_idx, x_range, colors_plot,
             xlabel, suptitle, fname):
    """metric_idx: 0=H, 1=C, 2=L"""

    fig, axes = setup_fig()

    for ax, (model_name, base_path) in zip(axes, MODELS.items()):
        style_ax(ax, model_name)
        handles = []

        for color in colors_plot:
            hex_c, lbl = COLOR_LINE[color]

            for segment, ls, alpha, lw in [("objecte", '-', 0.88, 1.2),
                                            ("fons",    ':', 0.70, 1.0)]:
                data = get_pixels(base_path, color, segment)
                arr  = data[metric_idx]
                if arr is None:
                    continue
                sorted_arr = np.sort(arr)
                cdf = np.arange(1, len(sorted_arr)+1) / len(sorted_arr)
                idx = np.linspace(0, len(sorted_arr)-1, 3000).astype(int)
                line, = ax.plot(sorted_arr[idx], cdf[idx], color=hex_c,
                                linewidth=lw, linestyle=ls, alpha=alpha)
                if segment == "objecte":
                    line.set_label(lbl)
                    handles.append(line)

        ax.set_xlabel(xlabel, fontsize=8, color='#444444', labelpad=4)
        ax.set_ylabel("Probabilitat acumulada", fontsize=8,
                      color='#444444', labelpad=4)
        ax.set_xlim(x_range)
        ax.set_ylim(0, 1)
        ax.grid(True, color='#F2F2F2', linewidth=0.5, zorder=0)

        if model_name == "sdxl_base" and handles:
            add_legend(ax, handles, loc='lower right')

    save_fig(fig, fname, suptitle)




# hue
make_cdf(0, x_range=(0, 360),
         colors_plot=COLORS_CROMATIC,
         xlabel="To $H^*$ (°)",
         suptitle="Distribucions within-image del To ($H^*$)",
         fname="opcioB_figura2_cdf_hue.png")

# chroma 
make_cdf(1, x_range=(0, 100),
         colors_plot=COLORS_ALL,
         xlabel="Croma $C^*$",
         suptitle="Distribucions within-image de la Croma ($C^*$)",
         fname="opcioB_figura3_cdf_chroma.png")

# lluminositat
make_cdf(2, x_range=(0, 100),
         colors_plot=COLORS_ALL,
         xlabel="Lluminositat $L^*$",
         suptitle="Distribucions within-image de la Lluminositat ($L^*$)",
         fname="opcioB_figura5_cdf_lightness.png")
