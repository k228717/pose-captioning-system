import cv2
import torch
import numpy as np
import mediapipe as mp
from model import PoseT5
from transformers import T5Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = PoseT5().to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
model.eval()

tokenizer = T5Tokenizer.from_pretrained("t5-small")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(1)

def normalize_pose(landmarks):
    coords = np.array([[lm.x, lm.y] for lm in landmarks])

    center = coords.mean(axis=0)
    coords = coords - center

    scale = np.linalg.norm(coords)
    if scale > 0:
        coords = coords / scale

    return coords.flatten()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark
        pose_vec = normalize_pose(lm)

        if len(pose_vec) >= 66:
            pose_vec = pose_vec[:66]

            pose_tensor = torch.tensor(pose_vec, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                embeds = model.input_projection(pose_tensor.unsqueeze(-1))

                ids = model.t5.generate(
                    inputs_embeds=embeds,
                    attention_mask=torch.ones(embeds.size()[:2]).to(device),
                    max_length=25,
                    num_beams=5,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )

            caption = tokenizer.decode(ids[0], skip_special_tokens=True)

            cv2.putText(frame, caption, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Pose Caption", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()