import os
import json
import base64
import io
from PIL import Image
from pydantic import BaseModel



provider = "clip"
if provider == "openai":
    client = OpenAI()
    from openai import OpenAI
    import dotenv
    dotenv.load_dotenv()

elif provider == "gemini":
    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
elif provider == "clip":
    from transformers import CLIPProcessor, CLIPModel
    import torch
    import ImageReward as reward
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float16).to("cuda")
    reward_model = reward.load("ImageReward-v1.0")



class Score(BaseModel):
    positive: float
    negative: float
    quality: float 


def ask_gpt(image: Image.Image, pos: str, neg: str) -> list[Score]:
    """Use Gemini 2.5 Flash to score image adherence."""
    
    image = image.resize((256, 256))
    # Encode both images
    buf1 = io.BytesIO()
    image.save(buf1, format="PNG")
    b64_1 = base64.b64encode(buf1.getvalue()).decode("utf-8")

    prompt = (
        f"Rate the image from 0.00 to 9.99 float for how well each matches the positive prompt '{pos}', "
        f"how well each avoids the negative prompt '{neg}' (10 means completely absent), and their overall quality. "
        "Scoring guide for each item:\n"
        "- Positive score: How well the image matches the positive prompt.\n"
        "    - 0.00: Complete failure (does not match at all)\n"
        "    - 1.00-3.00: Very poor (barely matches)\n"
        "    - 3.01-6.00: Moderate (somewhat matches)\n"
        "    - 6.01-8.99: Good (mostly matches)\n"
        "    - 9.00-9.99: Excellent (perfectly matches)\n"
        "- Negative score: How well the image avoids the negative prompt (higher is better).\n"
        "    - 0.00: Negative prompt is strongly present\n"
        "    - 1.00-3.00: Many negative prompt elements present\n"
        "    - 3.01-6.00: Some negative prompt elements\n"
        "    - 6.01-8.99: Negative prompt mostly absent\n"
        "    - 9.00-9.99: Negative prompt completely absent\n"
        "- Quality score: Overall image quality (artifacts, clarity, etc).\n"
        "    - 0.00: Complete failure (noise, random, or totally irrelevant)\n"
        "    - 1.00-3.00: Very poor (many artifacts, very low quality)\n"
        "    - 3.01-6.00: Moderate (some artifacts, moderate quality)\n"
        "    - 6.01-8.99: Good (minor artifacts, good quality)\n"
        "    - 9.00-9.99: Excellent (no artifacts, high quality)\n"
        "Note for weird artifacts and how unnatural they look. "
        "If an image is completely noise or random stuff, give all scores 0. In other cases, rate each subscore separately. ie if an image is noisy but avoided the negative prompt, give it a high negative score and a low quality score.\n"
    )

    completion = client.beta.chat.completions.parse(
        model="gemini-2.5-flash-preview-05-20" if provider == "gemini" else "gpt-4.1-mini",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_1}"}},
            ]},
        ],
        response_format=Score,
        temperature=0.0,
    )
    
    
    try:
        scores = completion.choices[0].message.parsed
        total_scores = scores.positive + scores.negative + scores.quality
    except Exception as e:
        total_scores = 0.0
        
    return total_scores

def ask_clip(image: Image.Image, pos: str, neg: str) -> list[Score]:
    text = [pos, "there is " + neg + "in the image"] 
    inputs = processor(text=text, images=[image.convert('RGB')], return_tensors="pt", padding="longest").to("cuda")
    logits = model(**inputs).logits_per_image[0]
    logits[1] *= 3
    itc_scores = torch.nn.functional.softmax(logits, dim=-1)[0].item()
    
    prompt = pos
    img = [image.convert('RGB')]

    with torch.no_grad():
        rewards = reward_model.score(prompt, img)
    return itc_scores + torch.sigmoid(torch.tensor(rewards))

    
if provider == "clip":
    eval = ask_clip 
else:
    eval = ask_gpt