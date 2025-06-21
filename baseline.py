import torch
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from module import Module
import prompts
from judge import eval 
import numpy as np
from PIL import Image
import wandb
import tqdm
import time
wandb.init(project="DPO") 

transformer = SD3Transformer2DModel.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16, subfolder="transformer")
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
pipe.transformer = transformer
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

print("len", len(prompts.negative_prompts))

scores = []

for idx in tqdm.tqdm(range(len(prompts.positive_prompts))):
    prompt = prompts.positive_prompts[idx]
    negative_prompt = prompts.negative_prompts[idx]
    image1 = pipe(
        prompt,
        negative_prompt=negative_prompt,
        width=512,
        height=512,
        num_inference_steps=16,
        guidance_scale=4,
    ).images[0]
    score = eval(image1, prompt, negative_prompt)
    scores.append(score)
    
print("mean score", np.mean(scores))