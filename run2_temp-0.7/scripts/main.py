# SamKash-Tolstoy blind test, run 2: model card settings (temperature 0.7).
# Prompts follow the model card's example format: "Write a short scene in the style of <work>: <scene>"
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
from peft import PeftModel

base_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
adpt_id = "salakash/SamKash-Tolstoy"  # or local folder

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32

tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype)
base.to(device)

model = PeftModel.from_pretrained(base, adpt_id)
model.config.use_cache = True

gen = pipeline("text-generation", model=model, tokenizer=tok, device=-1)

# Settings (same for every output)
TEMPERATURE = 0.7   # model card value
TOP_P = 0.9
MAX_NEW_TOKENS = 180   # blog post scene example value
OUTPUT_FILE = "inscope_test_outputs.csv"

SCENE = "a woman anxiously waits for a letter from a friend she hasn't seen in a long time."
WORKS = ["Anna Karenina", "Crime and Punishment", "The Lady with the Dog"]
prompts = {w: f"Write a short scene in the style of {w}: {SCENE}" for w in WORKS}

def run(prompt):
    return gen(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
               temperature=TEMPERATURE, top_p=TOP_P,
               return_full_text=False)[0]["generated_text"]

# Quick check: does switching the adapter off change the output?
set_seed(0); with_adapter = run(prompts[WORKS[0]])
set_seed(0)
with model.disable_adapter():
    without_adapter = run(prompts[WORKS[0]])
print("Adapter switch works:", with_adapter != without_adapter)
print(f"Settings: temperature={TEMPERATURE}, top_p={TOP_P}, max_new_tokens={MAX_NEW_TOKENS}")

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["test", "prompt", "condition", "sample", "text"])
    for work, prompt in prompts.items():
        for condition in ["Adapter", "Base"]:
            for sample in range(1, 4):
                set_seed(sample)
                if condition == "Base":
                    with model.disable_adapter():
                        text = run(prompt)
                else:
                    text = run(prompt)
                w.writerow(["In-scope", work, condition, sample, text])
                f.flush()
                print(f"done: {work} | {condition} | sample {sample}")
