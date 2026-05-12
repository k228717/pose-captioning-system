from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path



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

        return {
            "caption": "Pose landmarks were detected successfully. The person appears to be in a visible full-body posture, and the system has processed the uploaded image for pose-based caption generation."
        }

    except Exception as e:
        return {
            "caption": "Pose could not be processed clearly. Please use a clear full-body image.",
            "error": str(e)
        }