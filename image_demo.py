import cv2
import torch
import numpy as np
import mediapipe as mp
from model import PoseT5
from transformers import T5Tokenizer

IMAGE_PATH = "img.png"
MODEL_PATH = "checkpoints_improved/best_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = PoseT5().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

tokenizer = T5Tokenizer.from_pretrained("t5-small")

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


def normalize_pose(landmarks):
    coords = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    center = coords.mean(axis=0)
    coords = coords - center

    scale = np.linalg.norm(coords)
    if scale > 0:
        coords = coords / scale

    return coords.flatten()


def extract_pose_features(landmarks):
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_ankle = landmarks[27]
    right_ankle = landmarks[28]

    features = []

    if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
        features.append("both arms raised above the head")
    elif left_wrist.y < left_shoulder.y:
        features.append("left arm raised")
    elif right_wrist.y < right_shoulder.y:
        features.append("right arm raised")

    ankle_distance = abs(left_ankle.x - right_ankle.x)

    if ankle_distance < 0.08:
        features.append("legs close together")
    else:
        features.append("legs apart")

    if features:
        return "The person is standing with " + " and ".join(features) + "."

    return ""


image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image not found:", IMAGE_PATH)
    exit()

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose_detector:
    result = pose_detector.process(rgb)

if not result.pose_landmarks:
    print("No pose detected")
    exit()

mp_draw.draw_landmarks(
    image,
    result.pose_landmarks,
    mp_pose.POSE_CONNECTIONS,
    landmark_drawing_spec=mp_draw.DrawingSpec(
        color=(0, 0, 255),
        thickness=4,
        circle_radius=5
    ),
    connection_drawing_spec=mp_draw.DrawingSpec(
        color=(0, 255, 0),
        thickness=4
    )
)

pose_vec = normalize_pose(result.pose_landmarks.landmark)
pose_tensor = torch.tensor(pose_vec, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    embeds = model.input_projection(pose_tensor.unsqueeze(-1))

    generated_ids = model.t5.generate(
        inputs_embeds=embeds,
        attention_mask=torch.ones(embeds.size()[:2]).to(device),
        max_length=18,
        num_beams=3,
        repetition_penalty=1.5,
        length_penalty=0.7,
        early_stopping=True
    )

raw_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
rule_caption = extract_pose_features(result.pose_landmarks.landmark)
final_caption = rule_caption if rule_caption else raw_caption

print("\nRaw Model Caption:")
print(raw_caption)

print("\nRule Caption:")
print(rule_caption)

print("\nFinal Caption:")
print(final_caption)

# Bottom caption box
h, w, _ = image.shape
cv2.rectangle(image, (0, h - 95), (w, h), (0, 0, 0), -1)

cv2.putText(
    image,
    final_caption[:85],
    (15, h - 55),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0, 255, 0),
    2
)

cv2.imshow("Pose Caption Demo", image)

print("\nPress ESC to close the image window.")

while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()