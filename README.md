# Color-Statistics-in-AI-Generated-Images

Final degree project (TFG) at the Universitat Autònoma de Barcelona (UAB),
supervised by Alexandra Gomez Villa.

Replication and extension of Wang et al. (2025) *"Color statistics of images
created by generative AI"*, analyzing chromatic properties of AI-generated
images in the CIELAB color space.

## Overview

This project analyzes and compares the color statistics of images generated
by three text-to-image models (SDXL Base, SDXL Turbo, SD3) against a real
image baseline, across 4 object categories and 7 color prompts.

Color properties are measured in the **CIELAB** color space across three
dimensions: **hue (H*)**, **chroma (C*)**, and **lightness (L*)**, both at
the inter-image level (mean per image) and intra-image level (full pixel
distributions).


## Models analyzed

- SDXL Base: Open-source (Stability AI)
- SDXL Turbo: Open-source, distilled (Stability AI)
- Stable Diffusion 3: Open-source (Stability AI)
- Real images: Google Images (pre-2020))

## Real dataset used
https://www.kaggle.com/datasets/emmajuanico/color-statistics-in-ai-geneated-images/data

## Repository structure

├── ia_image_generation/              # Image generation scripts (SDXL Base, Turbo, SD3)

├── real_image_obtention/            # Real image obtention script

├── segmentation/                     # Segmentation pipeline (Grounded-SAM)

├── stadistics/

│   ├── inter-image/

│   │    ├── metriques_etapa1.py  # inter-image script

│   │    ├── model-color/  

│   │    │    ├── taules.py  # table generation for model-color

│   │    │    ├── elipses.py  # plot generation

│   │    ├── model-color-object/

│   │    │    ├── taules_objecte.py  # tables generation for model-color-object

│   ├── intra-image/    

│   ├── metriques etapa2.py

│   ├── plots_within.py 

│   ├── within_image_tables.py 

└── README.md


## Key methodological notes

- **Circular statistics** are used for hue (H*): mean and SD follow
  Mardia & Jupp (2000) to avoid the 0°/360° boundary problem.
- **Chroma clipping:** values of C* > 100 are clipped to 100
  (0.69% of pixels), consistent with Wang et al. (2025).
- **Total variance** is computed via PCA on Cartesian coordinates
  (X = C·cos H, Y = C·sin H) and reported as total explained variance.

Segmentation requires:
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [SAM ViT-H checkpoint](https://github.com/facebookresearch/segment-anything)


## Reference

Wang, Y. et al. (2025). *Color statistics of images created by generative AI*.
Journal of Vision, 25(1), B79.
