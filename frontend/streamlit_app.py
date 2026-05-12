import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import tempfile
import requests
import mediapipe as mp
from PIL import Image
import numpy as np
import os

BACKEND_URL = os.getenv("BACKEND_URL", "https://pose-captioning-system.onrender.com/predict")




st.set_page_config(
    page_title="AI Pose Captioning System",
    page_icon="🧠",
    layout="wide"
)


st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%);
    color: white;
}

.main-title {
    font-size: 52px;
    font-weight: 900;
    color: #ffffff;
    text-align: center;
    margin-bottom: 8px;
}

.sub-title {
    font-size: 20px;
    color: #cbd5e1;
    text-align: center;
    margin-bottom: 35px;
}

.glass-card {
    background: rgba(15, 23, 42, 0.86);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
}

.member-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
    border: 1px solid rgba(125, 211, 252, 0.25);
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 14px;
}

.member-name {
    font-size: 20px;
    font-weight: 800;
    color: #e0f2fe;
}

.member-roll {
    font-size: 15px;
    color: #94a3b8;
}

.caption-box {
    background: linear-gradient(135deg, rgba(8,47,73,0.95), rgba(15,23,42,0.95));
    border: 1px solid rgba(34,211,238,0.35);
    border-radius: 22px;
    padding: 26px;
    font-size: 20px;
    line-height: 1.8;
    color: #f8fafc;
    box-shadow: 0 12px 40px rgba(34,211,238,0.12);
}

.small-note {
    color: #94a3b8;
    font-size: 15px;
    line-height: 1.7;
}

.status-pill {
    display: inline-block;
    background: rgba(34, 197, 94, 0.14);
    border: 1px solid rgba(34, 197, 94, 0.35);
    color: #86efac;
    padding: 8px 14px;
    border-radius: 999px;
    font-weight: 700;
    margin-bottom: 16px;
}

.warning-pill {
    display: inline-block;
    background: rgba(251, 191, 36, 0.14);
    border: 1px solid rgba(251, 191, 36, 0.35);
    color: #fde68a;
    padding: 8px 14px;
    border-radius: 999px;
    font-weight: 700;
    margin-bottom: 16px;
}

div.stButton > button {
    background: linear-gradient(90deg, #06b6d4, #ec4899);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 12px 22px;
    font-size: 17px;
    font-weight: 800;
    box-shadow: 0 8px 24px rgba(236,72,153,0.25);
}

div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 30px rgba(6,182,212,0.30);
}
</style>
""", unsafe_allow_html=True)


def call_backend_with_image(image_path):
    with open(image_path, "rb") as file:
        files = {"file": file}
        response = requests.post(BACKEND_URL, files=files, timeout=120)

    if response.status_code != 200:
        return "Backend error occurred. Please check if FastAPI server is running."

    data = response.json()
    return data.get("caption", "No caption generated.")


def draw_pose_on_image(image_bgr):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as pose:

        results = pose.process(rgb)

    annotated = image_bgr.copy()

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(
                color=(0, 255, 255),
                thickness=3,
                circle_radius=4
            ),
            mp_drawing.DrawingSpec(
                color=(255, 0, 255),
                thickness=3
            )
        )

        detected = True

    else:
        detected = False

    return annotated, detected


class PoseTransformer(VideoTransformerBase):

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.latest_frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.pose.process(rgb)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(
                    color=(0, 255, 255),
                    thickness=3,
                    circle_radius=4
                ),
                self.mp_drawing.DrawingSpec(
                    color=(255, 0, 255),
                    thickness=3
                )
            )

        self.latest_frame = img.copy()

        return img


st.markdown(
    "<div class='main-title'>Real-Time AI Pose Captioning System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Live human pose detection, landmark visualization, and detailed natural-language pose caption generation</div>",
    unsafe_allow_html=True
)


left_col, right_col = st.columns([1, 2.1])


with left_col:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    st.markdown("## 👨‍💻 Group Members")

    st.markdown("""
    <div class='member-card'>
        <div class='member-name'>Javeria</div>
        <div class='member-roll'>Roll Number: 22K-8717</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='member-card'>
        <div class='member-name'>Zaina </div>
        <div class='member-roll'>Roll Number: 22K-0506</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='member-card'>
        <div class='member-name'>Taushar </div>
        <div class='member-roll'>Roll Number: 22K-4532</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ✨ System Pipeline")

    st.markdown("""
    <div class='small-note'>
    Camera/Image input → Pose landmark detection → 66 normalized pose values →
    trained PoseT5 caption model → hidden language refinement → final detailed caption.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


with right_col:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    st.markdown("## 📷 Pose Input Mode")

    mode = st.radio(
        "Choose input mode:",
        ["Upload Image", "Live Webcam"],
        horizontal=True
    )

    st.write("")

    if mode == "Upload Image":
        st.markdown("<span class='status-pill'>Upload Mode Active</span>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload a full-body pose image",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            annotated_bgr, detected = draw_pose_on_image(image_bgr)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            st.image(
                annotated_rgb,
                caption="Pose Landmark Detection Result",
                use_container_width=True
            )

            if detected:
                st.success("Pose landmarks detected successfully.")
            else:
                st.warning("Pose landmarks were not detected clearly. Try a clearer full-body image.")

            if st.button("✨ Generate Detailed Caption from Uploaded Image"):
                with st.spinner("Generating detailed pose caption..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                        cv2.imwrite(tmpfile.name, image_bgr)
                        caption = call_backend_with_image(tmpfile.name)

                st.markdown("## 🧠 Generated Detailed Caption")
                st.markdown(
                    f"""
                    <div class='caption-box'>
                    {caption}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    else:
        st.markdown("<span class='warning-pill'>Live Webcam Mode</span>", unsafe_allow_html=True)

        st.markdown("""
        <div class='small-note'>
        If webcam is not detected on this laptop, deploy or run the app on a device with a working camera.
        The upload mode can still be used for testing and demonstration.
        </div>
        """, unsafe_allow_html=True)

        ctx = webrtc_streamer(
            key="pose-webcam",
            video_transformer_factory=PoseTransformer,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )

        if st.button("✨ Generate Detailed Caption from Webcam Frame"):
            if ctx.video_transformer:
                frame = ctx.video_transformer.latest_frame

                if frame is not None:
                    with st.spinner("Capturing frame and generating caption..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                            cv2.imwrite(tmpfile.name, frame)
                            caption = call_backend_with_image(tmpfile.name)

                    st.markdown("## 🧠 Generated Detailed Caption")
                    st.markdown(
                        f"""
                        <div class='caption-box'>
                        {caption}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:
                    st.error("No webcam frame detected yet.")
            else:
                st.error("Webcam has not started. Please allow camera permission or use Upload Image mode.")

    st.markdown("</div>", unsafe_allow_html=True)