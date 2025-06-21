import torch
from pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from transformer_sd3 import SD3Transformer2DModel
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

def compute_loss(module, intermediate_latents, intermediate_prompt_embeds, pred, reward):
    loss = 0
    entropy = 0
    p = 0
    for i in range(len(intermediate_latents)):
        latents = intermediate_latents[i].cuda().float().unsqueeze(0)
        prompt_embeds = intermediate_prompt_embeds[i].cuda().float().unsqueeze(0)
        logistic = module(latents, prompt_embeds, i)
        prob = torch.softmax(logistic, dim=1)
        p += torch.gather(prob, 1, pred[i].cuda().long().unsqueeze(1)).squeeze(1).mean()
        # p += torch.max(prob, dim=1).values.mean()
        entropy += -torch.sum(prob * torch.log(prob + 1e-10), dim=1).mean()
        loss += torch.nn.functional.cross_entropy(logistic, pred[i].cuda().long(), reduction='mean', label_smoothing=0.01 if reward > 0 else 0.0)
    loss = loss / len(intermediate_latents)
    entropy = entropy / len(intermediate_latents)
    return loss * reward, entropy * 0, p / len(intermediate_latents)

module = Module().cuda()
optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)

print("len", len(prompts.negative_prompts))
losses = [] 
entropies = []
scores = []
ps = []
baseline = None

for epoch in range(500):
    for idx in tqdm.tqdm(range(len(prompts.positive_prompts))):
        seed = idx + epoch * len(prompts.positive_prompts)
        prompt = prompts.positive_prompts[idx]
        negative_prompt = prompts.negative_prompts[idx]
        
        image1 = pipe(
            prompt,
            negative_prompt=negative_prompt,
            width=512,
            height=512,
            num_inference_steps=16,
            guidance_scale=6,
            module=module,
            generator=torch.manual_seed(seed)
        ).images[0]

        import copy 
        intermediate_latents1 = [i.clone() for i in pipe.intermediate_latents]
        intermediate_prompt_embeds1 = [i.clone() for i in pipe.intermediate_prompt_embeds]
        pred1 = [i.clone() for i in pipe.pred]

        score = eval(image1, prompt, negative_prompt)
        
        loss, entropy, p = compute_loss(module, intermediate_latents1, intermediate_prompt_embeds1, pred1, score - baseline if baseline is not None else 0)
        (loss - entropy).backward()
        # loss.backward()
        if baseline is None:
            baseline = score
        else:
            baseline = 0.95 * baseline + 0.05 * score
        losses.append(loss.item())
        entropies.append(entropy.item())
        scores.append(score)
        ps.append(p.item())
        
        if (idx + 1) % 4 == 0: 
            optimizer.step()
            optimizer.zero_grad()
        
    wandb.log({
        "loss": np.mean(losses),
        "entropy": np.mean(entropies),
        "score": np.mean(scores),
        "p": np.mean(ps),
        "baseline": baseline,
    })
    losses = []
    entropies = []
    scores = []
    ps = []
            
    
    start_val_pipe1 = time.time()
    val_image1 = pipe(
            prompts.positive_prompts[8],
            negative_prompt=prompts.negative_prompts[8],
            width=512,
            height=512,
            num_inference_steps=16,
            guidance_scale=6,
            module=module, 
            temp=0.01,
            generator=torch.manual_seed(1989)
    ).images[0]
    time_val_pipe1 = time.time() - start_val_pipe1

    score = eval(val_image1, prompts.positive_prompts[8], prompts.negative_prompts[8])
    
    wandb.log({
        "val_image": wandb.Image(val_image1, caption=prompts.negative_prompts[8]),
        "val_score": score,
    })
    scheduler.step()
