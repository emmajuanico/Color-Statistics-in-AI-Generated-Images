from diffusers import AutoPipelineForText2Image
import torch
import os
import gc

BASE_DIR = ""
colors = ["red", "yellow", "green", "blue", "black", "gray", "white"]
objects = ["coffee mug", "chair", "bucket", "pen"]
IMAGES_PER_COMBINATION = 75

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe.enable_attention_slicing()

for obj in objects:
    for color in colors:
        path = f"{BASE_DIR}/{obj}/{color}"
        os.makedirs(path, exist_ok=True)

        for i in range(IMAGES_PER_COMBINATION):
            filename = f"{path}/{color}_{obj}_{i:03d}.png"

            if os.path.exists(filename):
                continue

            image = pipe(
                prompt=f"A single {color} {obj}",
                negative_prompt="cartoon, drawing, abstract, illustration, sketch, multiple objects, text, watermark",
                num_inference_steps=4,
                guidance_scale=0.0,
                height=1024,
                width=1024
            ).images[0]

            image.save(filename)


del pipe # alliberar memòria
torch.cuda.empty_cache()
gc.collect()
