from diffusers import StableDiffusion3Pipeline
import torch
import os

BASE_DIR = ""
colors = ["red", "yellow", "green", "blue", "black", "gray", "white"]
objects = ["coffee mug", "chair", "bucket", "pen"]
IMAGES_PER_COMBINATION = 75

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
    text_encoder_3=None,
    tokenizer_3=None
)

pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing() 

for obj in objects:
    for color in colors:
        path = f"{BASE_DIR}/{obj}/{color}"
        os.makedirs(path, exist_ok=True)

        for i in range(IMAGES_PER_COMBINATION):
            filename = f"{path}/{color}_{obj}_{i:03d}.png"

            if os.path.exists(filename): #saltar si ja s'havia generat
                continue

            image = pipe(
                prompt=f"A single {color} {obj}",
                negative_prompt="cartoon, drawing, abstract, illustration, sketch, multiple objects, text, watermark",
                num_inference_steps=28,
                guidance_scale=7.0,
                height=512,
                width=512
            ).images[0]

            image.save(filename)