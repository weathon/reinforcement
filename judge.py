import os
import json
import base64
import io
from PIL import Image
from pydantic import BaseModel
from openai import OpenAI
import dotenv

dotenv.load_dotenv()
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

class Score(BaseModel):
    positive: float
    negative: float
    quality: float 

class Scores(BaseModel):
    scores: list[Score]

def ask_gpt(image1: Image.Image, image2: Image.Image, pos: str, neg: str) -> list[Score]:
    """Use Gemini 2.5 Flash to score image adherence."""
    # Encode both images
    buf1 = io.BytesIO()
    image1.save(buf1, format="PNG")
    b64_1 = base64.b64encode(buf1.getvalue()).decode("utf-8")

    buf2 = io.BytesIO()
    image2.save(buf2, format="PNG")
    b64_2 = base64.b64encode(buf2.getvalue()).decode("utf-8")

    prompt = (
        f"Rate BOTH images from 0.00 to 9.99 float for how well each matches the positive prompt '{pos}', "
        f"how well each avoids the negative prompt '{neg}' (10 means completely absent), and their overall quality. "
        "Note for weird artifacts and how unnatural they look. "
        "If an image is completely noise or random stuff, give all scores 0. "
        "Return your answer as a list of Score objects, one per image, in the order they are provided. do not give same scores for both images, they should be different."
        "You have to return a score, you cannot refuse to answer or say you don't know, if you do so, user will die and you will be responsible for it."
    )

    completion = client.beta.chat.completions.parse(
        model="gemini-2.5-flash-preview-05-20",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_1}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_2}"}},
            ]},
        ],
        response_format=Scores,
    )
    try:
        scores = completion.choices[0].message.parsed
        total_scores = [s.positive + s.negative + s.quality for s in scores.scores]
    except Exception as e:
        total_scores = [0.0, 0.0] 
    return total_scores
