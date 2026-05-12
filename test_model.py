import torch
from dataset import PoseDataset
from model import PoseT5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = PoseDataset("final_training_normalized.csv")
sample = dataset[100]

model = PoseT5().to(device)

checkpoint_path = "checkpoints/best_model.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

pose = sample["pose"].unsqueeze(0).to(device)

with torch.no_grad():
    pose_embeds = model.input_projection(pose.unsqueeze(-1))

    generated_ids = model.t5.generate(
    inputs_embeds=pose_embeds,
    attention_mask=torch.ones(pose_embeds.size()[:2]).to(device),
    max_length=40,
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

print("Original Caption:")
print(original_caption)

print("\nGenerated Caption:")
print(generated_caption)