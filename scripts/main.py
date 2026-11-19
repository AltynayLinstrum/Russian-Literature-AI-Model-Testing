import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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

import csv
from transformers import set_seed

SCENE = "A man waits at a railway station for a train that is late."
prompts = {
    "Dostoevsky": f"Write a short scene in the style of Dostoevsky. {SCENE}",
    "Chekhov": f"Write a short scene in the style of Chekhov. {SCENE}",
    "Gogol": f"Write a short scene in the style of Gogol. {SCENE}",
}

def run(prompt):
    return gen(prompt, max_new_tokens=150, do_sample=True,
               temperature=0.8, top_p=0.9, return_full_text=False)[0]["generated_text"]

# Quick check: does switching the adapter off change the output?
set_seed(0); with_adapter = run(prompts["Dostoevsky"])
set_seed(0)
with model.disable_adapter():
    without_adapter = run(prompts["Dostoevsky"])
print("Adapter switch works:", with_adapter != without_adapter)

with open("inscope_test_outputs.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["test", "prompt", "condition", "sample", "text"])
    for author, prompt in prompts.items():
        for condition in ["Adapter", "Base"]:
            for sample in range(1, 4):
                set_seed(sample)
                if condition == "Base":
                    with model.disable_adapter():
                        text = run(prompt)
                else:
                    text = run(prompt)
                w.writerow(["In-scope", author, condition, sample, text])
                f.flush()
                print(f"done: {author} | {condition} | sample {sample}")