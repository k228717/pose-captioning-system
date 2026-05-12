import torch
from torch.utils.data import DataLoader, Subset
from dataset import PoseDataset
from model import PoseT5

# load dataset
dataset = PoseDataset("final_training_normalized.csv")

# only 5 samples
subset = Subset(dataset, list(range(5)))

loader = DataLoader(subset, batch_size=2, shuffle=True)

# model
model = PoseT5()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# training loop
for epoch in range(20):
    total_loss = 0

    for batch in loader:
        pose = batch["pose"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = model(pose, input_ids, attention_mask)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")