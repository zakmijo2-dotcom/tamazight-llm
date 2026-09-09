import os
import json
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_num_threads(8)

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
data_path = "/data/data/com.termux/files/home/tamazight_llm/tamazight_train.jsonl"
output_dir = "/data/data/com.termux/files/home/tamazight_llm/tamazight-smollm-135m"
os.makedirs(output_dir, exist_ok=True)

class LoRALinear(nn.Module):
    def __init__(self, linear, r=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.randn(r, linear.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, r))

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def merge(self):
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.linear.weight.data.add_(delta)
        return self.linear

print("1. Loading Model & Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, low_cpu_mem_usage=True)

for p in model.parameters():
    p.requires_grad = False

lora_params = []
for layer in model.model.layers[20:]:
    layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=8, alpha=16)
    layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=8, alpha=16)
    lora_params.extend([
        layer.self_attn.q_proj.lora_A, layer.self_attn.q_proj.lora_B,
        layer.self_attn.v_proj.lora_A, layer.self_attn.v_proj.lora_B
    ])

optimizer = torch.optim.AdamW(lora_params, lr=1e-4, weight_decay=0.01)

# Build system-prompt backed instruction pairs
sys_prefix = "<|im_start|>system\nKečč d amussnaw n tmaziɣt. Tettmeslayeḍ s tmaziɣt (Taqbaylit) s ugemmay alatini.<|im_end|>\n"

raw_samples = []
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        p = d.get("prompt", "")
        if p:
            # prepend system prompt
            full_prompt = sys_prefix + p
            raw_samples.append(full_prompt)
        if len(raw_samples) >= 500:
            break

print(f"Loaded {len(raw_samples)} formatted training samples with system prompt.")

model.train()
print("\n2. Training on Tamazight Latin text...")
start_time = time.time()
step = 0
max_steps = 60

for text in raw_samples:
    enc = tokenizer(text, max_length=56, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"]
    labels = input_ids.clone()

    optimizer.zero_grad()
    out = model(input_ids=input_ids, labels=labels)
    loss = out.loss

    if torch.isnan(loss) or torch.isinf(loss):
        continue

    loss.backward()
    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
    optimizer.step()

    step += 1
    if step % 10 == 0 or step == 1:
        elapsed = time.time() - start_time
        print(f"Step {step:2d}/{max_steps} | Loss: {loss.item():.4f} | Speed: {elapsed/step:.2f}s/step")

    if step >= max_steps:
        break

print(f"\nTraining completed in {time.time() - start_time:.1f}s!")

# Merge LoRA
for layer in model.model.layers[20:]:
    layer.self_attn.q_proj = layer.self_attn.q_proj.merge()
    layer.self_attn.v_proj = layer.self_attn.v_proj.merge()

print("Merged LoRA weights successfully.")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")
EOF
