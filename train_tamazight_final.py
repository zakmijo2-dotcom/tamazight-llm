import os
import json
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_num_threads(8)

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
data_path = "tamazight_train.jsonl" if os.path.exists("tamazight_train.jsonl") else "/data/data/com.termux/files/home/tamazight_llm/tamazight_train.jsonl"
output_dir = "tamazight-smollm-135m" if not os.path.exists("/data/data/com.termux") else "/data/data/com.termux/files/home/tamazight_llm/tamazight-smollm-135m"
os.makedirs(output_dir, exist_ok=True)

class LoRALinear(nn.Module):
    def __init__(self, linear, r=4, alpha=4):
        super().__init__()
        self.linear = linear
        self.scaling = alpha / r
        # Proper LoRA init: A with small gaussian, B with exact zeros
        self.lora_A = nn.Parameter(torch.randn(r, linear.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, r))

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def merge(self):
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.linear.weight.data.add_(delta)
        return self.linear

print("=== 1. Loading Base Model & Tokenizer ===")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
    low_cpu_mem_usage=True
)

# Freeze base model
for p in model.parameters():
    p.requires_grad = False

# Attach LoRA to top layers (layers 26 to 29)
lora_layers = []
lora_params = []
for layer in model.model.layers[26:]:
    layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=4, alpha=4)
    layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=4, alpha=4)
    lora_layers.extend([layer.self_attn.q_proj, layer.self_attn.v_proj])
    lora_params.extend([
        layer.self_attn.q_proj.lora_A, layer.self_attn.q_proj.lora_B,
        layer.self_attn.v_proj.lora_A, layer.self_attn.v_proj.lora_B
    ])

print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable LoRA parameters: {sum(p.numel() for p in lora_params):,} across {len(lora_layers)} projections")

# Optimizer
optimizer = torch.optim.AdamW(lora_params, lr=1e-4, weight_decay=0.01)

# Read training prompts
samples = []
with open(data_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 150:
            break
        d = json.loads(line)
        p = d.get("prompt", "")
        if p and len(p) > 10:
            samples.append(p)

print(f"Loaded {len(samples)} diverse Tamazight samples.")

print("\n=== 2. Starting Training Loop ===")
start_time = time.time()
step = 0
total_steps = 25

model.train()
for prompt in samples:
    enc = tokenizer(prompt, max_length=40, truncation=True, return_tensors="pt")
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
    if step % 5 == 0 or step == 1:
        elapsed = time.time() - start_time
        print(f"Step {step:2d}/{total_steps} | Loss: {loss.item():.4f} | Speed: {elapsed/step:.2f}s/step | Logits max: {out.logits.max().item():.1f}")

    if step >= total_steps:
        break

print(f"\nTraining completed in {time.time() - start_time:.1f}s!")

# Merge LoRA weights
print("\n=== 3. Merging LoRA Weights into Base Model ===")
for layer in model.model.layers[26:]:
    layer.self_attn.q_proj = layer.self_attn.q_proj.merge()
    layer.self_attn.v_proj = layer.self_attn.v_proj.merge()

print("All LoRA layers merged cleanly into base model weights.")

# Save final model
print(f"\n=== 4. Saving Merged Model to {output_dir} ===")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("Model and Tokenizer successfully saved!")
EOF
