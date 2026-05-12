import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

HEIGHT_THR = 0.025
HORIZONTAL_THR = 0.025
FEET_APART_THR = 0.08
BEND_THR = 8


def to_np(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def angle_3pts(a, b, c):
    ba = a - b
    bc = c - b

    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9

    cosv = np.dot(ba, bc) / denom
    cosv = np.clip(cosv, -1, 1)

    return float(np.degrees(np.arccos(cosv)))


def extract_pose_relations(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Image not found for correction layer.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as pose:

        result = pose.process(image_rgb)

    if not result.pose_landmarks:
        raise ValueError("No pose detected for correction layer.")

    lms = result.pose_landmarks.landmark

    def L(i):
        return to_np(lms[i])

    LS = L(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    RS = L(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)

    LE = L(mp_pose.PoseLandmark.LEFT_ELBOW.value)
    RE = L(mp_pose.PoseLandmark.RIGHT_ELBOW.value)

    LW = L(mp_pose.PoseLandmark.LEFT_WRIST.value)
    RW = L(mp_pose.PoseLandmark.RIGHT_WRIST.value)

    LH = L(mp_pose.PoseLandmark.LEFT_HIP.value)
    RH = L(mp_pose.PoseLandmark.RIGHT_HIP.value)

    LK = L(mp_pose.PoseLandmark.LEFT_KNEE.value)
    RK = L(mp_pose.PoseLandmark.RIGHT_KNEE.value)

    LA = L(mp_pose.PoseLandmark.LEFT_ANKLE.value)
    RA = L(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

    left_knee_angle = angle_3pts(LH, LK, LA)
    right_knee_angle = angle_3pts(RH, RK, RA)

    left_elbow_angle = angle_3pts(LS, LE, LW)
    right_elbow_angle = angle_3pts(RS, RE, RW)

    feet_distance = float(np.linalg.norm(LA[:2] - RA[:2]))

    both_hands_above_shoulders = (
        LW[1] < LS[1] and RW[1] < RS[1]
    )

    both_arms_raised = (
        LW[1] < LS[1] and RW[1] < RS[1]
    )

    knees_straight = (
        left_knee_angle > 160 and right_knee_angle > 160
    )

    feet_close = feet_distance <= FEET_APART_THR

    return {
        "left_knee_angle": round(left_knee_angle, 2),
        "right_knee_angle": round(right_knee_angle, 2),

        "left_elbow_angle": round(left_elbow_angle, 2),
        "right_elbow_angle": round(right_elbow_angle, 2),

        "feet_distance": round(feet_distance, 3),

        "arms": (
            "both arms raised above the shoulders"
            if both_arms_raised
            else "arms are not both raised"
        ),

        "hands": (
            "both hands are above shoulder level"
            if both_hands_above_shoulders
            else "hands are not both above shoulder level"
        ),

        "knees": (
            "both knees appear mostly straight"
            if knees_straight
            else "one or both knees appear bent"
        ),

        "feet": (
            "feet are close together"
            if feet_close
            else "feet are spaced apart"
        ),

        "body": "body is upright and facing mostly forward"
    }


def build_corrected_caption(raw_caption, relations):
    parts = []

    parts.append("The person is standing upright and facing mostly forward.")

    if relations["feet"] == "feet are close together":
        parts.append("The feet are close together.")
    else:
        parts.append("The feet are spaced apart.")

    if relations["arms"] == "both arms raised above the shoulders":
        parts.append("Both arms are raised above the shoulders.")
    else:
        parts.append("Both arms are lowered beside the body.")

    if relations["hands"] == "both hands are above shoulder level":
        parts.append("Both hands are positioned above shoulder level.")
    else:
        parts.append("Both hands are positioned below shoulder level.")

    if relations["knees"] == "both knees appear mostly straight":
        parts.append("Both knees appear mostly straight.")
    else:
        parts.append("One or both knees appear slightly bent.")

    parts.append("Overall, the pose shows a stable standing posture.")

    return " ".join(parts)