import os
import json
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_num_threads(8)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
data_path = "data/multilingual_chat.jsonl" if os.path.exists("data/multilingual_chat.jsonl") else "tamazight_train.jsonl"
output_dir = "tamazight_qwen_finetuned"
os.makedirs(output_dir, exist_ok=True)

print("1. Loading Tokenizer and Model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
    low_cpu_mem_usage=True
)

# Freeze all layers except the last transformer block, norm, and lm_head
for p in model.parameters():
    p.requires_grad = False

for p in model.model.layers[-1].parameters():
    p.requires_grad = True

for p in model.model.norm.parameters():
    p.requires_grad = True

for p in model.lm_head.parameters():
    p.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {total_params:,} | Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")

# 2. Prepare Dataset
class TamazightDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_samples=1000, max_len=64):
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

print("2. Building training DataLoader...")
dataset = TamazightDataset(data_path, tokenizer, max_samples=800, max_len=64)
loader = DataLoader(dataset, batch_size=2, shuffle=True)
print(f"Loaded {len(dataset)} samples for training ({len(loader)} batches).")

# 3. Optimizer & Training Loop
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4,
    weight_decay=0.01
)

model.train()
print("\n3. Starting Training on Tamazight Latin Corpus...")
start_time = time.time()
step = 0
max_steps = 40

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
        print(f"Step {step:2d}/{max_steps} | Loss: {loss.item():.4f} | Elapsed: {elapsed:.1f}s")

    if step >= max_steps:
        break

print(f"\nTraining completed in {time.time() - start_time:.1f}s!")

# 4. Save the fine-tuned model
print("\n4. Saving fine-tuned model...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Fine-tuned model and tokenizer successfully saved to: {output_dir}")
EOF
