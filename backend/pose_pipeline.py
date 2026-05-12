import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import cv2
import numpy as np
import torch
import mediapipe as mp
from transformers import T5Tokenizer

from model import PoseT5
from backend.language_refiner import generate_refined_caption


MODEL_PATH = BASE_DIR / "checkpoints" / "best_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mp_pose = mp.solutions.pose

model = None
tokenizer = None


def load_model_once():
    global model, tokenizer

    if model is not None and tokenizer is not None:
        return model, tokenizer

    print("Loading model...")

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    model = PoseT5("t5-small")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    return model, tokenizer


def extract_pose_points(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Image not found or cannot be read.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as pose:

        result = pose.process(image_rgb)

    if not result.pose_landmarks:
        raise ValueError("No human pose detected in the image.")

    points = []

    for lm in result.pose_landmarks.landmark:
        points.append(lm.x)
        points.append(lm.y)

    points = np.array(points, dtype=np.float32)

    if len(points) != 66:
        raise ValueError(f"Expected 66 pose values, got {len(points)}")

    return points


def generate_model_caption_from_points(pose_points):
    model, tokenizer = load_model_once()

    pose_tensor = torch.tensor(
        pose_points,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        pose_tensor = pose_tensor.unsqueeze(-1)

        pose_embeds = model.input_projection(
            pose_tensor
        )

        attention_mask = torch.ones(
            pose_embeds.size()[:2]
        ).to(DEVICE)

        output_ids = model.t5.generate(
            inputs_embeds=pose_embeds,
            attention_mask=attention_mask,
            max_new_tokens=80,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    caption = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    return caption


def run_pipeline(image_path):

    pose_points = extract_pose_points(
        image_path
    )

    model_caption = generate_model_caption_from_points(
        pose_points
    )

    final_caption = generate_refined_caption(
        model_caption,
        pose_points.tolist()
    )

    return {
        "final_caption": final_caption,
        "model_caption": model_caption,
        "pose_points": pose_points.tolist()
    }


if __name__ == "__main__":

    test_image = BASE_DIR / "img5.png"

    result = run_pipeline(test_image)

    print("\nFinal Caption:")
    print(result["final_caption"])