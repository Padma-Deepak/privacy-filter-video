"""
app.py — Context-Aware Privacy Filtering System
Renders a side-by-side results page (original vs. processed) in the browser.
Supports both still images and short video clips.
"""

import os
import base64
import uuid

from flask import Flask, render_template, request, abort
from detector import process_image, process_video

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

MIME_TYPES = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
    "bmp": "image/bmp",  "webp": "image/webp",
    "mp4": "video/mp4",  "webm": "video/webm", "mov": "video/quicktime",
    "avi": "video/x-msvideo", "mkv": "video/x-matroska",
}

for d in (UPLOADS_DIR, OUTPUTS_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB upload cap


def _allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _is_video(ext):
    return ext in ALLOWED_VIDEO_EXTENSIONS


def _to_data_url(path, ext):
    mime = MIME_TYPES.get(ext, "application/octet-stream")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        abort(400, description="No file part in the request.")
    file = request.files["image"]
    if file.filename == "":
        abort(400, description="No file selected.")
    if not _allowed(file.filename):
        abort(400, description="Unsupported type. Please upload an image (JPEG, PNG, BMP, WEBP) or a short video (MP4, MOV, AVI, MKV, WEBM).")

    ext = file.filename.rsplit(".", 1)[1].lower()
    is_video = _is_video(ext)
    upload_path = os.path.join(UPLOADS_DIR, f"{uuid.uuid4().hex}.{ext}")
    file.save(upload_path)

    output_path = None
    try:
        if is_video:
            result = process_video(upload_path, OUTPUTS_DIR, MODELS_DIR)
        else:
            result = process_image(upload_path, OUTPUTS_DIR, MODELS_DIR)
        output_path = result["output_path"]
        out_ext = "mp4" if is_video else ext  # process_video always emits mp4

        # Encode both files for embedding — then delete temp files
        original_data  = _to_data_url(upload_path, ext)
        processed_data = _to_data_url(output_path, out_ext)
    except Exception as exc:
        abort(500, description=f"Processing failed: {exc}")
    finally:
        for p in (upload_path, output_path):
            if p:
                try: os.remove(p)
                except OSError: pass

    return render_template(
        "result.html",
        original=original_data,
        processed=processed_data,
        faces=result["faces_found"],
        plates=result["plates_found"],
        screens=result["screens_found"],
        ext=out_ext,
        media_type="video" if is_video else "image",
        frames=result.get("frames_processed"),
        truncated=result.get("truncated", False),
    )


@app.errorhandler(400)
def bad_request(e):
    return render_template("index.html", error=e.description), 400

@app.errorhandler(413)
def too_large(e):
    return render_template("index.html", error="File is too large. Please upload something under 200MB."), 413

@app.errorhandler(500)
def server_error(e):
    return render_template("index.html", error=e.description), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
