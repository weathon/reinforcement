import torch
from pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from transformer_sd3 import SD3Transformer2DModel
from module import Module
import prompts
from judge import ask_gpt 
import numpy as np
from PIL import Image
import wandb
import tqdm
import time

wandb.init(project="DPO")

transformer = SD3Transformer2DModel.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16, subfolder="transformer")
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe.transformer = transformer
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

def compute_term(module, intermediate_latents, intermediate_prompt_embeds, pred):
    term = 0
    entropy = 0
    for i in range(len(intermediate_latents)):
        latents = intermediate_latents[i].cuda().float().unsqueeze(0)
        prompt_embeds = intermediate_prompt_embeds[i].cuda().float().unsqueeze(0)
        logistic = module(latents, prompt_embeds, i)
        prob = torch.softmax(logistic, dim=1) 
        entropy += -torch.sum(prob * torch.log(prob + 1e-10), dim=1).mean()
        prob = prob.reshape(prob.shape[1], -1) 
        pred_ = pred[i].reshape(-1)
        selected_prob = torch.gather(prob, 0, pred_.unsqueeze(0).long()).squeeze(0)
        selected_log_prob = torch.log(selected_prob + 1e-10)
        term += torch.mean(selected_log_prob)
    return term, entropy

module = Module().cuda()
optimizer = torch.optim.AdamW(module.parameters(), lr=5e-5, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)

print("len", len(prompts.negative_prompts))
for epoch in range(1000):
    for idx in tqdm.tqdm(range(len(prompts.positive_prompts))):
        seed = idx + epoch * len(prompts.positive_prompts)
        prompt = prompts.positive_prompts[idx]
        negative_prompt = prompts.negative_prompts[idx]
        
        # Time tracing for the first pipe call
        start_pipe1 = time.time()
        image1 = pipe(
            prompt,
            negative_prompt=negative_prompt,
            width=512,
            height=512,
            num_inference_steps=16,
            guidance_scale=4,
            module=module,
            generator=torch.manual_seed(seed)
        ).images[0]
        time_pipe1 = time.time() - start_pipe1

        import copy 
        intermediate_latents1 = [i.clone() for i in pipe.intermediate_latents]
        intermediate_prompt_embeds1 = [i.clone() for i in pipe.intermediate_prompt_embeds]
        pred1 = [i.clone() for i in pipe.pred]

        # Time tracing for the second pipe call
        start_pipe2 = time.time()
        image2 = pipe(
            prompt,
            negative_prompt=negative_prompt,
            width=512,
            height=512,
            num_inference_steps=16,
            guidance_scale=4,
            module=module,
            generator=torch.manual_seed(seed + 1)
        ).images[0]
        time_pipe2 = time.time() - start_pipe2

        # Time tracing for ask_gpt call
        start_ask = time.time()
        score1, score2 = ask_gpt(image1, image2, prompt, negative_prompt)
        time_ask = time.time() - start_ask

        # print(f"Scores for image1: {score1}, image2: {score2}")

        intermediate_latents2 = [i.clone() for i in pipe.intermediate_latents]
        intermediate_prompt_embeds2 = [i.clone() for i in pipe.intermediate_prompt_embeds]
        pred2 = [i.clone() for i in pipe.pred]

        pipe.pred[-1].shape

        term1, entropy1 = compute_term(module, intermediate_latents1, intermediate_prompt_embeds1, pred1)
        term2, entropy2 = compute_term(module, intermediate_latents2, intermediate_prompt_embeds2, pred2)

        # if score1 > score2:
        #     sign = 1
        # else:
        #     sign = -1

        # loss = - torch.log(torch.sigmoid(sign * (term1 - term2)))
        policy_loss = (- term1 * (score1 - 23) / 30 - term2 * (score2 - 23) / 30) 
        entropy_loss = (- entropy1 - entropy2)
        loss = policy_loss# + entropy_loss
        loss.backward()
        wandb.log({
            "loss": loss.item(),
            "image": wandb.Image(Image.fromarray(np.concatenate([np.array(image1), np.array(image2)], axis=1)), caption=negative_prompt),
            "p1": np.exp(term1.item()),
            "p2": np.exp(term2.item()),
            "score1": score1,
            "score2": score2,
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "time_pipe1": time_pipe1,
            "time_pipe2": time_pipe2,
            "time_ask": time_ask
        })
        # if (idx % (len(prompts.positive_prompts) // 2)) == (len(prompts.positive_prompts) // 2 - 1):
        if idx % 32 == 31:
            # print("updated")
            optimizer.step()
            optimizer.zero_grad()
    
    # Validation section: time tracing for both pipe calls and ask_gpt
    start_val_pipe1 = time.time()
    val_image1 = pipe(
            prompts.positive_prompts[8],
            negative_prompt=prompts.negative_prompts[8],
            width=512,
            height=512,
            num_inference_steps=16,
            guidance_scale=4,
            module=module,
            temp=0.0001,
            generator=torch.manual_seed(1989)
    ).images[0]
    time_val_pipe1 = time.time() - start_val_pipe1

    start_val_pipe2 = time.time()
    val_image2 = pipe(
            prompts.positive_prompts[8],
            negative_prompt=prompts.negative_prompts[8],
            width=512,
            height=512,
            num_inference_steps=16,
            guidance_scale=4,
            module=module,
            temp=0.0001,
            generator=torch.manual_seed(1990)
    ).images[0]
    time_val_pipe2 = time.time() - start_val_pipe2
    
    start_val_ask = time.time()
    score1, score2 = ask_gpt(val_image1, val_image2, prompts.positive_prompts[8], prompts.negative_prompts[8])
    time_val_ask = time.time() - start_val_ask
    
    wandb.log({
        "val_image": wandb.Image(Image.fromarray(np.concatenate([np.array(val_image1), np.array(val_image2)], axis=1)), caption=prompts.negative_prompts[8]),
        "val_score1": score1,
        "val_score2": score2,
        "time_val_pipe1": time_val_pipe1,
        "time_val_pipe2": time_val_pipe2,
        "time_val_ask": time_val_ask
    })
scheduler.step()
