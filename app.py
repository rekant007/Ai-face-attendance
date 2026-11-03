import cv2
import mediapipe as mp

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from engine.liveness import BlinkDetector


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
HAAR_PATH = MODELS_DIR / "haarcascade_frontalface_default.xml"
LANDMARKER_PATH = MODELS_DIR / "face_landmarker.task"


for path in [YUNET_PATH, SFACE_PATH, HAAR_PATH, LANDMARKER_PATH]:
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")


yunet = cv2.FaceDetectorYN.create(
    str(YUNET_PATH),
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)

sface = cv2.FaceRecognizerSF.create(
    str(SFACE_PATH),
    ""
)

haarcascade = cv2.CascadeClassifier(str(HAAR_PATH))

if haarcascade.empty():
    raise RuntimeError("Haar Cascade initialization failed!")


landmarker_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(LANDMARKER_PATH)
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)

landmarker = vision.FaceLandmarker.create_from_options(
    landmarker_options
)

blink_detector = BlinkDetector()

print("All AI components initialized successfully!")