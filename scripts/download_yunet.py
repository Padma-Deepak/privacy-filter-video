"""Fetch the MIT-licensed OpenCV Zoo YuNet weights; never bundle weights in Git."""
from download_models import _fetch

REVISION = "47534e27c9851bb1128ccc0102f1145e27f23f98"
MODEL = "face_detection_yunet_2023mar.onnx"
SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

if __name__ == "__main__":
    _fetch(MODEL, f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{REVISION}/models/face_detection_yunet/{MODEL}", SHA256)
