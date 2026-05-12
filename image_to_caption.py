import cv2
import numpy as np
import torch
import mediapipe as mp
from transformers import T5Tokenizer
from model import PoseT5

MODEL_PATH = "checkpoints/best_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mp_pose = mp.solutions.pose


def extract_66_points(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found. Please check image path.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        result = pose.process(image_rgb)

    if not result.pose_landmarks:
        raise ValueError("No pose detected in image.")

    points = []

    for lm in result.pose_landmarks.landmark:
        points.append(lm.x)
        points.append(lm.y)

    points = np.array(points, dtype=np.float32)

    if len(points) != 66:
        raise ValueError(f"Expected 66 values, got {len(points)}")

    return points


def load_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    model = PoseT5("t5-small")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model, tokenizer


def generate_caption(image_path):
    model, tokenizer = load_model()

    pose_points = extract_66_points(image_path)

    pose_tensor = torch.tensor(pose_points, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pose_tensor = pose_tensor.unsqueeze(-1)
        pose_embeds = model.input_projection(pose_tensor)

        attention_mask = torch.ones(pose_embeds.size()[:2]).to(DEVICE)

        output_ids = model.t5.generate(
            inputs_embeds=pose_embeds,
            attention_mask=attention_mask,
            max_length=50,
            num_beams=4,
            early_stopping=True
        )

    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return caption


if __name__ == "__main__":
    image_path = "img.png"

    caption = generate_caption(image_path)

    print("\nGenerated Caption:")
    print(caption)