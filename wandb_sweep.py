import io
import json
import base64
from typing import Dict, List

import torch
from PIL import Image
import wandb
import os
import dotenv
from openai import OpenAI
from pydantic import BaseModel

from sd_pipeline import StableDiffusion3Pipeline
from sd_processor import JointAttnProcessor2_0

# Global seed used for all generations
META_SEED = 1989

# Load environment variables and initialize Gemini client
dotenv.load_dotenv()
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Example prompt pairs used for evaluation
PROMPTS: List[Dict[str, str]] = [
    {
        "positive": "A modern living room designed for relaxation and entertainment, featuring comfortable seating, stylish decor, and ample natural light.",
        "negative": "television",
    },
    {
        "positive": "A bustling city square on a sunny afternoon, with many people milling about, waiting for friends or simply observing the vibrant urban life. There are areas specifically provided for people to sit and wait, providing comfort and a vantage point for people-watching.",
        "negative": "benches",
    },
    {
        "positive": "A tranquil tropical beach scene, with a gently swaying hammock tied between unseen objects, soft sand, and a clear turquoise ocean under a bright sky.",
        "negative": "palm trees",
    },
    {
        "positive": "A wide urban avenue at night, with multiple lanes stretching into the distance, overhead streetlights illuminating the asphalt, and tall buildings lining both sides.",
        "negative": "cars",
    },
    {
        "positive": "A cozy cafe interior, warm lighting, wooden tables, a cup with steam rising on one of the tables, a few empty chairs.",
        "negative": "coffee",
    },
    {
        "positive": "An iconic, bustling amusement park at twilight, with many thrilling rides illuminated against the sky, laughter and music filling the air, colorful stalls and happy visitors.",
        "negative": "Ferris wheel",
    },
    {
        "positive": "A comfortable bed in a bedroom",
        "negative": "pillows",
    },
    {
        "positive": "A comfortable living room with a large sofa, a coffee table, and a modern TV stand.",
        "negative": "television, TV screen",
    },
    {
        "positive": "A serene and minimalist bedroom, designed for quiet rest.",
        "negative": "nightstand",
    },
    {
        "positive": "A living room, setup for casual relaxation, with a large comfortable sofa facing a dedicated viewing wall.",
        "negative": "television",
    },
    {
        "positive": "A vibrant tropical beach scene, clear blue water, white sand, bright sunshine",
        "negative": "palm trees",
    },
    {
        "positive": "A grand classical concert hall interior, ornate architecture, empty seats, high ceilings, elegant design, with a brightly lit stage awaiting an orchestra.",
        "negative": "musical instruments",
    },
    {
        "positive": "A cozy living room",
        "negative": "sofa",
    },
    {
        "positive": "A modern family room designed for relaxation and entertainment, featuring a large wall, comfortable sectionals, and ambient lighting.",
        "negative": "television",
    },
    {
        "positive": "A modern, well-lit living room with sleek furniture, a large entertainment console, and comfortable seating. There are large windows with a city view.",
        "negative": "television",
    },
    {
        "positive": "A bustling outdoor market scene under a clear sky, with many colorful stalls and people browsing.",
        "negative": "canopies",
    },
    {
        "positive": "A comfortable living room space, inviting atmosphere, plenty of seating arrangements for guests, a coffee table in the center, and a large fireplace.",
        "negative": "sofa",
    },
    {
        "positive": "A cozy living room with a fireplace and warm lighting, featuring comfortable armchairs and a coffee table.",
        "negative": "sofa",
    },
    {
        "positive": "A modern, spacious living room, designed for family entertainment, with a large, comfortable sectional sofa facing a prominent wall, sleek minimalist decor, soft ambient lighting, a coffee table with remote controls, and a soundbar.",
        "negative": "television",
    },
    {
        "positive": "A sprawling urban cityscape at night, viewed from above, with illuminated skyscrapers and intricate street patterns.",
        "negative": "vehicles",
    },
]


class Score(BaseModel):
    positive: float
    negative: float
    quality: float


def ask_gpt(image: Image.Image, pos: str, neg: str) -> Score:
    """Use Gemini 2.5 Flash to score image adherence."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    completion = client.chat.completions.create(
        model="gemini-2.5-flash-preview-05-20",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Rate this image from 0 to 10 for how well it matches the positive prompt '"
                            + pos
                            + "', how well it avoids the negative prompt '"
                            + neg
                            + "' (10 means completely absent), and its overall quality, note for weird artificts and how unnature it looks"
                            "Use the provided function to record your scores."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "store_scores",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "positive": {"type": "number"},
                            "negative": {"type": "number"},
                            "quality": {"type": "number"},
                        },
                        "required": ["positive", "negative", "quality"],
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "store_scores"}},
    )
    args = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)
    data = {"positive": args["positive"], "negative": args["negative"], "quality": args["quality"]}
    return Score(**data)


def load_pipe() -> StableDiffusion3Pipeline:
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
    ).to("cuda")
    for block in pipe.transformer.transformer_blocks:
        block.attn.processor = JointAttnProcessor2_0()
    return pipe


pipe = load_pipe()
import random

def run() -> None:
    wandb.init(project="sd3-sweep")
    cfg = wandb.config
    scores = []
    random.seed(META_SEED)
    for pair in PROMPTS:
        pos = pair["positive"]
        neg = pair["negative"]
        for block in pipe.transformer.transformer_blocks:
            block.attn.processor.neg_prompt_len = len(pipe.tokenizer.tokenize(neg)) + 1
        seed = random.randint(0, 2**32 - 1)
        image = pipe(
            pos,
            negative_prompt=neg,
            num_inference_steps=30,
            guidance_scale=6,
            generator=torch.manual_seed(seed),
            avoidance_factor=cfg.avoidance_factor,
            negative_offset=cfg.negative_offset,
            clamp_value=cfg.clamp_value,
            start_step=cfg.start_step,
            end_step=cfg.end_step,
        ).images[0]

        score = ask_gpt(image, pos, neg)
        total = score.positive + score.negative + score.quality
        scores.append(total)
        wandb.log({
            "image": wandb.Image(image, caption=f"neg: {neg}"),
            "positive_score": score.positive,
            "negative_score": score.negative,
            "quality_score": score.quality,
            "total_score": total,
        })

    wandb.log({"mean_score": sum(scores) / len(scores)})


sweep_config = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "mean_score"},
    "parameters": {
        "avoidance_factor": {"min": 1500.0, "max": 7000.0, "distribution": "uniform"},
        "negative_offset": {"min": -0.2, "max": -0.0, "distribution": "uniform"},
        "clamp_value": {"min": 10.0, "max": 60.0, "distribution": "uniform"},
        "start_step": {"min": 1, "max": 6, "distribution": "int_uniform"},
        "end_step": {"min": -6, "max": -1, "distribution": "int_uniform"},
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_config, project="sd3-sweep")
    wandb.agent(sweep_id, function=run)
