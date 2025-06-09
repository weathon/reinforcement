import torch
from pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from transformer_sd3 import SD3Transformer2DModel
from module import Module
import prompts
from judge import ask_gpt 
import numpy as np
from PIL import Image
import wandb
wandb.init(project="DPO")

transformer = SD3Transformer2DModel.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16, subfolder="transformer")
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe.transformer = transformer
pipe = pipe.to("cuda")

def compute_term(module, intermediate_latents, intermediate_prompt_embeds, pred):
    term = 0
    for i in range(len(intermediate_latents)):
        latents = intermediate_latents[i].cuda().float().unsqueeze(0)
        prompt_embeds = intermediate_prompt_embeds[i].cuda().float().unsqueeze(0)
        logistic = module(latents, prompt_embeds)
        prob = torch.softmax(logistic, dim=1) 
        prob = prob.reshape(prob.shape[1], -1) 
        pred_ = pred[i].reshape(-1)
        selected_prob = torch.gather(prob, 0, pred_.unsqueeze(0).long()).squeeze(0)
        selected_log_prob = torch.log(selected_prob + 1e-10)
        term += torch.mean(selected_log_prob)
    return term

module = Module().cuda()
optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4)

for epoch in range(500):
    for idx in range(len(prompts.positive_prompts)):
        optimizer.zero_grad()
        prompt = prompts.positive_prompts[idx]
        negative_prompt = prompts.negative_prompts[idx]
        image1 = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=16,
            guidance_scale=4,
            module=module
        ).images[0] 

        import copy 
        intermediate_latents1 = copy.deepcopy(pipe.intermediate_latents)
        intermediate_prompt_embeds1 = copy.deepcopy(pipe.intermediate_prompt_embeds)
        pred1 = copy.deepcopy(pipe.pred)

        image2 = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=16,
            guidance_scale=4,
            module=module
        ).images[0] 

        score1, score2 = ask_gpt(image1, image2, prompt, negative_prompt) 
        print(f"Scores for image1: {score1}, image2: {score2}")

        intermediate_latents2 = copy.deepcopy(pipe.intermediate_latents)
        intermediate_prompt_embeds2 = copy.deepcopy(pipe.intermediate_prompt_embeds)
        pred2 = copy.deepcopy(pipe.pred)


        pipe.pred[-1].shape


        term1 = compute_term(module, intermediate_latents1, intermediate_prompt_embeds1, pred1)
        term2 = compute_term(module, intermediate_latents2, intermediate_prompt_embeds2, pred2)


        if score1 > score2: 
            sign = 1
        else:
            sign = -1

        loss = - torch.log(torch.sigmoid(sign * (term1 - term2))) 
        loss.backward()
        wandb.log({"loss": loss.item(), "reward": sign * (term1 - term2), "image": wandb.Image(Image.fromarray(np.concatenate([np.array(image1), np.array(image2)], axis=1))), "p1": np.exp(term1.item()), "p2": np.exp(term2.item()), "term1": term1.item(), "term2": term2.item()})
        optimizer.step()
    
