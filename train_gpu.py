import torch
from torch.utils.data import DataLoader
from dataset import PoseDataset
from model import PoseT5
import os

# ================= CONFIG =================
CSV_PATH = "final_training_normalized.csv"
BATCH_SIZE = 8
LR = 5e-5
EPOCHS = 10
SAVE_DIR = "checkpoints"
LOG_FILE = "training_log_continue.txt"
RESUME_PATH = os.path.join(SAVE_DIR, "best_model.pt")

os.makedirs(SAVE_DIR, exist_ok=True)

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= DATA =================
dataset = PoseDataset(CSV_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ================= MODEL =================
model = PoseT5().to(device)

if os.path.exists(RESUME_PATH):
    print(f"Loading previous best model from: {RESUME_PATH}")
    model.load_state_dict(torch.load(RESUME_PATH, map_location=device))
else:
    print("No previous best_model.pt found. Training from scratch.")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_loss = float("inf")

# ================= LOG FILE =================
log_f = open(LOG_FILE, "a", encoding="utf-8")
log_f.write("\n\n========== CONTINUED TRAINING STARTED ==========\n")

# ================= TRAIN LOOP =================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for step, batch in enumerate(loader):
        pose = batch["pose"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(pose, input_ids, attention_mask)
        loss = outputs.loss

        if torch.isnan(loss):
            print("NaN detected — stopping training")
            log_f.write("NaN detected\n")
            break

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        if step % 50 == 0:
            msg = f"[Continue Epoch {epoch+1} | Step {step}] Loss: {loss.item():.4f}"
            print(msg)
            log_f.write(msg + "\n")
            log_f.flush()

    avg_loss = total_loss / len(loader)

    msg = f"\nContinue Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f}\n"
    print(msg)
    log_f.write(msg)
    log_f.flush()

    checkpoint_path = os.path.join(SAVE_DIR, f"continued_model_epoch_{epoch+1}.pt")
    torch.save(model.state_dict(), checkpoint_path)

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_path = os.path.join(SAVE_DIR, "best_model.pt")
        torch.save(model.state_dict(), best_path)
        print("Best model updated!")
        log_f.write("Best model updated\n")
        log_f.flush()

log_f.close()

print("Continued training complete!")