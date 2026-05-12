from dataset import PoseDataset

dataset = PoseDataset("final_training_normalized.csv")

print("Dataset size:", len(dataset))

sample = dataset[0]

print("\nPose shape:", sample["pose"].shape)
print("Input IDs shape:", sample["input_ids"].shape)
print("Attention mask shape:", sample["attention_mask"].shape)

print("\nFirst 5 pose values:", sample["pose"][:5])
print("First 10 token ids:", sample["input_ids"][:10])