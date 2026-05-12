from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path

from backend.pose_pipeline import run_pipeline

app = FastAPI(title="Pose Captioning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def home():
    return {
        "message": "Pose Captioning Backend is running"
    }


@app.post("/predict")
async def predict_pose_caption(file: UploadFile = File(...)):
    try:

        file_path = UPLOAD_DIR / file.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = run_pipeline(file_path)

        return {
            "caption": result["final_caption"]
        }

    except Exception as e:

        return {
            "caption": "Pose could not be detected clearly. Please use a clear full-body image."
        }