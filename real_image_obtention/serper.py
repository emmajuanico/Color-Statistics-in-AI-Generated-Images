import requests
import os
import time
from PIL import Image
from io import BytesIO

API_KEY = ''
BASE_DIR = ""
TOTAL_FINAL = 120
SEARCH_LIMIT = 100

OBJECTS = ["bucket", "pen", "coffee mug", "chair"]
COLORS = ["black", "blue", "gray", "green", "red", "white", "yellow"]


def download_dataset():
    url = "https://google.serper.dev/images"
    headers = {'X-API-KEY': API_KEY, 'Content-Type': 'application/json'}
    download_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }

    for obj in OBJECTS:
        for color in COLORS:
            folder_path = os.path.join(BASE_DIR, obj, color)
            os.makedirs(folder_path, exist_ok=True)

            existing_images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
            num_existing = len(existing_images)

            if num_existing >= TOTAL_FINAL:
                print(f"Saltant {color} {obj}: Ja hi ha {num_existing} imatges.")
                continue

            needed = TOTAL_FINAL - num_existing

            query = f'real life photo of a {color} {obj}'

            payload = {
                "q": query,
                "num": SEARCH_LIMIT,
                "page": 1,
                "gl": "us",
                "tbs": "cdr:1,cd_max:12/31/2019" #filtre pre-ia
            }

            try:
                response = requests.post(url, headers=headers, json=payload, timeout=15)
                results = response.json()
            except Exception as e:
                print(f"Error en la petició API: {e}")
                continue

            success_count = 0
            image_list = results.get("images", [])

            next_idx = num_existing + 1

            for img_info in image_list:
                if success_count >= needed:
                    break

                img_url = img_info["imageUrl"]
                try:
                    r = requests.get(img_url, headers=download_headers, timeout=7)
                    r.raise_for_status()

                    image = Image.open(BytesIO(r.content))
                    image = image.convert("RGB")

                    file_name = f"{color}_{obj}_{next_idx}.jpg"
                    file_path = os.path.join(folder_path, file_name)

                    image.save(file_path, "JPEG")
                    success_count += 1
                    next_idx += 1

                except Exception:
                    continue

            print(f"Finalitzat: {color} {obj} (Total ara: {num_existing + success_count})")
            time.sleep(1.5)


if __name__ == "__main__":
    download_dataset()