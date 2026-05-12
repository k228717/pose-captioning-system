import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import PoseDataset
from model import PoseT5
from collections import Counter
import math

CSV_PATH = "final_training_normalized.csv"
MODEL_PATH = "checkpoints_improved/best_model.pt"
OUTPUT_CSV = "evaluation_generated_captions.csv"
METRICS_FILE = "evaluation_metrics.txt"

BATCH_SIZE = 8
EVAL_SAMPLES = 500   # full dataset ke liye None kar sakti ho, but slow hoga

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = PoseDataset(CSV_PATH)
model = PoseT5().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ================= LOSS EVALUATION =================
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

total_loss = 0
count = 0

with torch.no_grad():
    for batch in loader:
        pose = batch["pose"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(pose, input_ids, attention_mask)
        loss = outputs.loss

        total_loss += loss.item()
        count += 1

avg_loss = total_loss / count
print("Average Loss:", avg_loss)

# ================= BLEU HELPERS =================
def tokenize(text):
    return text.lower().replace(".", "").replace(",", "").split()

def ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def bleu_score(reference, candidate, max_n=4):
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    if len(cand_tokens) == 0:
        return 0.0

    precisions = []

    for n in range(1, max_n + 1):
        ref_counts = ngram_counts(ref_tokens, n)
        cand_counts = ngram_counts(cand_tokens, n)

        overlap = 0
        total = sum(cand_counts.values())

        for gram, count in cand_counts.items():
            overlap += min(count, ref_counts.get(gram, 0))

        # smoothing
        precision = (overlap + 1) / (total + 1)
        precisions.append(precision)

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)

    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)

    if cand_len > ref_len:
        bp = 1
    else:
        bp = math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0

    return bp * geo_mean

# ================= GENERATION EVALUATION =================
results = []
bleu_scores = []

num_samples = EVAL_SAMPLES if EVAL_SAMPLES is not None else len(dataset)
num_samples = min(num_samples, len(dataset))

print(f"Generating captions for {num_samples} samples...")

for i in range(num_samples):
    sample = dataset[i]

    pose = sample["pose"].unsqueeze(0).to(device)

    with torch.no_grad():
        pose_embeds = model.input_projection(pose.unsqueeze(-1))

        generated_ids = model.t5.generate(
            inputs_embeds=pose_embeds,
            attention_mask=torch.ones(pose_embeds.size()[:2]).to(device),
            max_length=30,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    generated_caption = dataset.tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    original_caption = dataset.tokenizer.decode(
        sample["input_ids"],
        skip_special_tokens=True
    )

    bleu = bleu_score(original_caption, generated_caption)
    bleu_scores.append(bleu)

    results.append({
        "sample_index": i,
        "original_caption": original_caption,
        "generated_caption": generated_caption,
        "bleu_score": bleu
    })

    if i % 50 == 0:
        print(f"Done {i}/{num_samples}")

avg_bleu = sum(bleu_scores) / len(bleu_scores)

# ================= SAVE RESULTS =================
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

with open(METRICS_FILE, "w", encoding="utf-8") as f:
    f.write("FINAL MODEL EVALUATION\n")
    f.write("======================\n")
    f.write(f"Model Path: {MODEL_PATH}\n")
    f.write(f"Dataset: {CSV_PATH}\n")
    f.write(f"Total Dataset Rows: {len(dataset)}\n")
    f.write(f"Evaluated Samples for BLEU: {num_samples}\n\n")
    f.write(f"Average Loss: {avg_loss:.4f}\n")
    f.write(f"Average BLEU Score: {avg_bleu:.4f}\n")

print("\nEvaluation complete!")
print("Average Loss:", round(avg_loss, 4))
print("Average BLEU Score:", round(avg_bleu, 4))
print("Saved:", OUTPUT_CSV)
print("Saved:", METRICS_FILE)