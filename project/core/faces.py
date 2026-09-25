"""Local YuNet face detection. Missing weights fail explicitly, never use Haar."""
from pathlib import Path
import threading
import cv2

_lock = threading.RLock()
_models = {}


def detect_faces(frame, models_dir, confidence=0.8):
    path = str(Path(models_dir) / "face_detection_yunet_2023mar.onnx")
    if not Path(path).is_file():
        raise ValueError("YuNet face model missing. Run python scripts/download_yunet.py first.")
    height, width = frame.shape[:2]
    with _lock:
        model = _models.get(path)
        if model is None:
            model = cv2.FaceDetectorYN.create(path, "", (width, height), confidence, 0.3, 5000)
            _models[path] = model
        model.setInputSize((width, height))
        model.setScoreThreshold(float(confidence))
        _, faces = model.detect(frame)
    boxes = []
    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4]
            left, top = max(0, round(float(x))), max(0, round(float(y)))
            right, bottom = min(width, round(float(x+w))), min(height, round(float(y+h)))
            if right > left and bottom > top:
                boxes.append((left, top, right-left, bottom-top))
    return boxes
