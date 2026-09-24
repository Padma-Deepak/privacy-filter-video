"""
app.py — Context-Aware Privacy Filtering System
Renders a side-by-side results page (original vs. processed) in the browser.
Supports both still images and short video clips.
"""

import os
import base64
import subprocess
import uuid

from flask import Flask, render_template, request, abort
from detector import process_image
from core.video import process_video_streaming

# Containers/MIME types browsers reliably play inline via a <video> tag —
# notably NOT video/quicktime (.mov), which is the actual cause of the
# "Original panel doesn't load" bug found on a real iPhone upload: Chrome
# does not reliably play video/quicktime inline, especially via a data: URI.
# .mov/.avi/.mkv uploads get remuxed (not re-encoded — fast, lossless) to a
# clean single-video+audio MP4 for the PREVIEW only; the actual uploaded
# file and the processing pipeline are untouched.
WEB_PLAYABLE_VIDEO_EXTENSIONS = {"mp4", "webm"}

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


def _env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# Off and localhost-only by default — Flask's debug mode runs the Werkzeug
# interactive debugger, which allows arbitrary code execution from the
# browser if it's ever reachable from a real network. Both are opt-in via
# environment variables for local development only.
DEBUG = _env_flag("FLASK_DEBUG", False)
HOST = os.environ.get("FLASK_HOST", "127.0.0.1")
PORT = int(os.environ.get("FLASK_PORT", "5000"))

# Unlimited by default (PHASES.md Phase 1) — set for the hosted demo (Phase 6)
# to bound processing time/output size, e.g. MAX_CLIP_SECONDS=15.
_max_clip_seconds_raw = os.environ.get("MAX_CLIP_SECONDS", "").strip()
MAX_CLIP_SECONDS = float(_max_clip_seconds_raw) if _max_clip_seconds_raw else None

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


def _preview_data_url(path, ext, is_video):
    """
    Build the data: URL used for the browser preview only.

    For video containers browsers don't reliably play inline (.mov/.avi/.mkv),
    remux to a temporary clean MP4 first — a stream copy (-c copy), so it's
    fast and lossless, not a re-encode. Falls back to the raw file if the
    remux fails for any reason, so a preview quirk never turns into a 500.
    """
    if not is_video or ext in WEB_PLAYABLE_VIDEO_EXTENSIONS:
        return _to_data_url(path, ext)

    remuxed_path = f"{path}_preview.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-map", "0:v:0", "-map", "0:a:0?",
             "-c", "copy", "-movflags", "+faststart", "-loglevel", "error", remuxed_path],
            check=True, timeout=60,
        )
        return _to_data_url(remuxed_path, "mp4")
    except Exception:
        app.logger.warning("Preview remux failed for %s, falling back to raw file", path, exc_info=True)
        return _to_data_url(path, ext)
    finally:
        if os.path.exists(remuxed_path):
            os.remove(remuxed_path)


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
            result = process_video_streaming(
                upload_path, OUTPUTS_DIR, MODELS_DIR, max_clip_seconds=MAX_CLIP_SECONDS,
            )
        else:
            result = process_image(upload_path, OUTPUTS_DIR, MODELS_DIR)
        output_path = result["output_path"]
        out_ext = "mp4" if is_video else ext  # process_video always emits mp4

        # Encode both files for embedding — then delete temp files
        original_data  = _preview_data_url(upload_path, ext, is_video)
        processed_data = _to_data_url(output_path, out_ext)
    except Exception:
        app.logger.exception("Processing failed for upload %s", upload_path)
        abort(500, description="Processing failed. Please try a different file.")
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
        warnings=result.get("warnings", []),
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
    app.run(debug=DEBUG, host=HOST, port=PORT)
