import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_num_threads(8)

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
data_path = "tamazight_train.jsonl" if os.path.exists("tamazight_train.jsonl") else "/data/data/com.termux/files/home/tamazight_llm/tamazight_train.jsonl"
output_dir = "tamazight_smollm_finetuned" if not os.path.exists("/data/data/com.termux") else "/data/data/com.termux/files/home/tamazight_llm/tamazight_smollm_finetuned"
os.makedirs(output_dir, exist_ok=True)

class LoRALinear(nn.Module):
    def __init__(self, linear, r=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        # Standard LoRA init: A with small gaussian, B with zero
        self.lora_A = nn.Parameter(torch.randn(r, linear.in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, r))

    def forward(self, x):
        base_out = self.linear(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    def merge(self):
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.linear.weight.data.add_(delta)
        return self.linear

print("1. Loading Tokenizer and Model...")
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

# Attach LoRA to q_proj and v_proj in all layers
lora_layers = []
for layer in model.model.layers:
    layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=8, alpha=16)
    layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=8, alpha=16)
    lora_layers.extend([layer.self_attn.q_proj, layer.self_attn.v_proj])

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {total_params:,} | Trainable LoRA params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

# 2. Build Dataset
class TamazightDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_samples=600, max_len=64):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                data = json.loads(line)
                prompt = data.get("prompt", "")
                if prompt:
                    self.samples.append(prompt)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

print("2. Loading training batch...")
dataset = TamazightDataset(data_path, tokenizer, max_samples=400, max_len=64)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-4,
    weight_decay=0.01
)

model.train()
print("\n3. Training LoRA on Tamazight Latin Corpus...")
start_time = time.time()
step = 0
max_steps = 35

for batch in loader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    optimizer.zero_grad()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    loss = outputs.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    step += 1
    if step % 5 == 0 or step == 1:
        elapsed = time.time() - start_time
        print(f"Step {step:2d}/{max_steps} | Loss: {loss.item():.4f} | Speed: {elapsed/step:.2f}s/step")

    if step >= max_steps:
        break

print(f"\nTraining completed successfully in {time.time() - start_time:.1f}s!")

# 4. Merge LoRA adapters into base model weights
print("\n4. Merging LoRA adapters into model weights...")
for layer in model.model.layers:
    layer.self_attn.q_proj = layer.self_attn.q_proj.merge()
    layer.self_attn.v_proj = layer.self_attn.v_proj.merge()

print("5. Saving merged fine-tuned model...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Final model and tokenizer successfully saved to: {output_dir}")
EOF
