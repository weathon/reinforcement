import os
import json
from openai import OpenAI
from pydantic import BaseModel
import dotenv

dotenv.load_dotenv()
client = OpenAI()

class PosNegResponse(BaseModel):
    pos: str
    neg: str

class PromptExtractionResponse(BaseModel):
    positive_prompt: str
    negative_prompt: str

def main():
    completion = client.beta.chat.completions.parse(
        model="gpt4.1",
        messages=[
            {"role": "system", "content": "Extract positive and negative information from the user's message."},
            {"role": "user", "content": "Hello from Ollama!"},
        ],
        response_format=PosNegResponse,
    )
    return completion.choices[0].message.parsed

def extract_prompts(long_description):
    extraction_completion = client.beta.chat.completions.parse(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": (
                "Given a description, extract:\n"
                "1. A positive prompt: make the given description long and descriptive and removing the negative part, do not shorten, paraphrase, or add information.\n"
                "2. A negative prompt as a string listing items to avoid, separated by commas (e.g., \"fish, tree\"). "
                "Do NOT use negated phrases like 'no fish', 'not on the table', 'not drinking' just list the items, it should be a list of nouns to avoid and NOT a description.\n"
                "Example: Given: A metallic call and a fabric notebook are on the same desk, the notebook are not red. Answer: Positive prompt: A metallic call and a fabric notebook are on the same desk. bla bla bla (make it more descriptive) Negative prompt: red notebook.\n"
            )},
            {"role": "user", "content": long_description}, 
        ],
        response_format=PromptExtractionResponse,
    )
    return extraction_completion.choices[0].message.parsed

def get_negation_jsonl_contents(directory):
    negation_contents = []
    for fname in os.listdir(directory):
        if fname.endswith('.jsonl') and 'negation' in fname and "texture" not in fname:
            print(fname)
            fpath = os.path.join(directory, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        negation_contents.append(line)
    return negation_contents
import tqdm
if __name__ == "__main__":
    negation_list = get_negation_jsonl_contents('TIIF-Bench/data/test_prompts')
    extracted_results = []
    for item in tqdm.tqdm(negation_list):
        obj = json.loads(item)
        long_desc = obj.get("short_description")
        if long_desc:
            prompts = extract_prompts(long_desc)
            result = {
                "short_description": long_desc,
                "positive_prompt": prompts.positive_prompt,
                "negative_prompt": prompts.negative_prompt
            }
            extracted_results.append(result)
            
    print(extracted_results)
    with open('extracted_prompts.json', 'w', encoding='utf-8') as f:
        json.dump(extracted_results, f, ensure_ascii=False, indent=4)