import torch
from torch.utils.data import DataLoader
from dataset import PoseDataset
from model import PoseT5
import os

# ================= CONFIG =================
CSV_PATH = "final_training.csv"   # same improved dataset
BATCH_SIZE = 8
LR = 3e-5   # 🔥 lower LR for better stability
EPOCHS = 5
SAVE_DIR = "checkpoints_improved"
LOG_FILE = "training_log_improved.txt"

os.makedirs(SAVE_DIR, exist_ok=True)

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= DATA =================
dataset = PoseDataset(CSV_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ================= MODEL =================
model = PoseT5().to(device)

# 🔥 Load OLD best model (IMPORTANT)
model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_loss = float("inf")

# ================= LOG FILE =================
log_f = open(LOG_FILE, "w")

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
            print("NaN detected — stopping")
            break

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if step % 100 == 0:
            msg = f"[Improved Epoch {epoch+1} | Step {step}] Loss: {loss.item():.4f}"
            print(msg)
            log_f.write(msg + "\n")

    avg_loss = total_loss / len(loader)

    msg = f"\nImproved Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f}\n"
    print(msg)
    log_f.write(msg)

    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pt"))

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
        print("🔥 Improved best model saved!")

log_f.close()
print("✅ Improved training complete!")