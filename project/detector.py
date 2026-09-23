"""
detector.py — Context-Aware Privacy Filtering System

Detection strategy:
  • Faces          → Gaussian blur   (Haar Cascade + DNN SSD)
  • License plates → Black mask      (Haar Cascade + contour)
  • Screens / phones / laptops → Pixelation  (YOLOv8)

Performance fixes:
  • All models (Haar cascades, DNN net, YOLO) loaded ONCE as module-level
    singletons — never re-read from disk per request.
  • Face, plate and screen detection run in PARALLEL via ThreadPoolExecutor.

Entry points:
  • process_image(...) — single image, returns an output file path + counts.
  • process_video(...) — short video clip, runs the same per-frame pipeline
    (process_frame) over every frame and re-encodes an MP4 (video only, no
    audio track).
"""

import os
import uuid
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# ──────────────────────────────────────────────────────────────────────────────
# YOLO class IDs (COCO) we treat as "screens / devices"
# ──────────────────────────────────────────────────────────────────────────────
SCREEN_CLASSES = {
    62: "tv",
    63: "laptop",
    67: "cell phone",
}

# ──────────────────────────────────────────────────────────────────────────────
# Module-level singletons  — loaded ONCE at import time, reused forever
# ──────────────────────────────────────────────────────────────────────────────

# Haar: frontal face
_haar_face_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
_haar_face_clf  = cv2.CascadeClassifier(_haar_face_path)

# Haar: license plate
_haar_plate_path = os.path.join(cv2.data.haarcascades, "haarcascade_russian_plate_number.xml")
_haar_plate_clf  = (
    cv2.CascadeClassifier(_haar_plate_path)
    if os.path.isfile(_haar_plate_path) else None
)

# DNN SSD face net  — populated lazily so missing model files don't crash startup
_dnn_net = None

def _get_dnn(models_dir: str):
    """Load the Caffe SSD face net once; return None if files are absent."""
    global _dnn_net
    if _dnn_net is not None:
        return _dnn_net
    proto = os.path.join(models_dir, "deploy.prototxt")
    model = os.path.join(models_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    if not os.path.isfile(proto) or not os.path.isfile(model):
        return None
    try:
        _dnn_net = cv2.dnn.readNetFromCaffe(proto, model)
    except Exception:
        _dnn_net = None
    return _dnn_net

# YOLO
_yolo_model = None

def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO("yolov8n.pt")
        except Exception:
            _yolo_model = None
    return _yolo_model

# Eagerly warm up YOLO at import time
_get_yolo()


# ──────────────────────────────────────────────────────────────────────────────
# NMS helper
# ──────────────────────────────────────────────────────────────────────────────

def _nms(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = areas.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        iou = (w * h) / (areas[i] + areas[order[1:]] - w * h)
        order = order[np.where(iou <= overlap_thresh)[0] + 1]
    return [tuple(map(int, boxes[i])) for i in keep]


# ──────────────────────────────────────────────────────────────────────────────
# Face detection  (Haar singleton + DNN SSD singleton)
# ──────────────────────────────────────────────────────────────────────────────

def _detect_faces_haar(gray):
    det = _haar_face_clf.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=4,
        minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE
    )
    return [tuple(int(v) for v in d) for d in det] if len(det) else []


def _detect_faces_dnn(image, models_dir, conf=0.5):
    net = _get_dnn(models_dir)
    if net is None:
        return []
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    dets = net.forward()
    boxes = []
    for i in range(dets.shape[2]):
        if float(dets[0, 0, i, 2]) < conf:
            continue
        x1 = int(dets[0, 0, i, 3] * w)
        y1 = int(dets[0, 0, i, 4] * h)
        bw = max(0, int(dets[0, 0, i, 5] * w) - x1)
        bh = max(0, int(dets[0, 0, i, 6] * h) - y1)
        if bw > 0 and bh > 0:
            boxes.append((x1, y1, bw, bh))
    return boxes


def detect_faces(image, gray, models_dir):
    return _nms(_detect_faces_haar(gray) + _detect_faces_dnn(image, models_dir))


# ──────────────────────────────────────────────────────────────────────────────
# License-plate detection  (Haar singleton + contour)
# ──────────────────────────────────────────────────────────────────────────────

def _detect_plates_haar(gray):
    if _haar_plate_clf is None:
        return []
    det = _haar_plate_clf.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 10)
    )
    return [tuple(int(v) for v in d) for d in det] if len(det) else []


def _detect_plates_contour(gray):
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 1500:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) < 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if h > 0 and 2.0 <= w / h <= 5.5:
            boxes.append((x, y, w, h))
    return boxes


def detect_plates(gray):
    return _nms(_detect_plates_haar(gray) + _detect_plates_contour(gray))


# ──────────────────────────────────────────────────────────────────────────────
# Screen / device detection  (YOLOv8 singleton)
# ──────────────────────────────────────────────────────────────────────────────

def detect_screens(image, conf=0.4):
    model = _get_yolo()
    if model is None:
        return []
    try:
        results = model(image, verbose=False, conf=conf)
        boxes = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in SCREEN_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Filter functions
# ──────────────────────────────────────────────────────────────────────────────

def _clamp_box(x, y, w, h, img_w, img_h):
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    return x1, y1, x2, y2


def apply_gaussian_blur(image, x, y, w, h, kernel=(99, 99), sigma=30):
    """Smooth blur — used for faces."""
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = _clamp_box(x, y, w, h, img_w, img_h)
    if x2 <= x1 or y2 <= y1:
        return
    image[y1:y2, x1:x2] = cv2.GaussianBlur(image[y1:y2, x1:x2], kernel, sigma)


def apply_pixelation(image, x, y, w, h, blocks=15):
    """Pixelation / mosaic — used for screens and devices."""
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = _clamp_box(x, y, w, h, img_w, img_h)
    if x2 <= x1 or y2 <= y1:
        return
    roi = image[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh == 0 or rw == 0:
        return
    small = cv2.resize(roi, (max(1, rw // blocks), max(1, rh // blocks)),
                        interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    image[y1:y2, x1:x2] = pixelated


def apply_black_mask(image, x, y, w, h):
    """Solid black rectangle — used for license plates."""
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = _clamp_box(x, y, w, h, img_w, img_h)
    if x2 <= x1 or y2 <= y1:
        return
    image[y1:y2, x1:x2] = 0


# ──────────────────────────────────────────────────────────────────────────────
# Single-frame pipeline — shared by both the image and video entry points
# ──────────────────────────────────────────────────────────────────────────────

def process_frame(image, models_dir: str, executor: ThreadPoolExecutor = None):
    """
    Run all three detectors on a single BGR frame and apply the matching
    context-aware filter to a copy of it.

    Pass a shared `executor` when calling this repeatedly (e.g. once per video
    frame) to avoid the overhead of spinning up a new thread pool every call.

    Returns (output_image, counts) where counts is
    {"faces": int, "plates": int, "screens": int}.
    """
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if executor is not None:
        futures = {
            executor.submit(detect_faces,   image, gray, models_dir): "faces",
            executor.submit(detect_plates,  gray):                     "plates",
            executor.submit(detect_screens, image):                    "screens",
        }
        results_map = {futures[f]: f.result() for f in as_completed(futures)}
    else:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(detect_faces,   image, gray, models_dir): "faces",
                pool.submit(detect_plates,  gray):                     "plates",
                pool.submit(detect_screens, image):                    "screens",
            }
            results_map = {futures[f]: f.result() for f in as_completed(futures)}

    face_boxes   = results_map["faces"]
    plate_boxes  = results_map["plates"]
    screen_boxes = results_map["screens"]

    for (x, y, w, h) in face_boxes:
        apply_gaussian_blur(output, x, y, w, h)   # Gaussian blur

    for (x, y, w, h) in plate_boxes:
        apply_black_mask(output, x, y, w, h)       # Black mask

    for (x, y, w, h) in screen_boxes:
        apply_pixelation(output, x, y, w, h)       # Pixelation

    counts = {
        "faces":   len(face_boxes),
        "plates":  len(plate_boxes),
        "screens": len(screen_boxes),
    }
    return output, counts


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point — images
# ──────────────────────────────────────────────────────────────────────────────

def process_image(input_path: str, outputs_dir: str, models_dir: str) -> dict:
    """
    Context-aware privacy filtering pipeline for a single image.

    Detectors run in PARALLEL (ThreadPoolExecutor) so total time ≈ slowest
    single detector rather than sum of all three.

    Returns dict:
        output_path   : str
        faces_found   : int
        plates_found  : int
        screens_found : int
    """
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    output, counts = process_frame(image, models_dir)

    os.makedirs(outputs_dir, exist_ok=True)
    ext = os.path.splitext(input_path)[1].lower() or ".jpg"
    output_path = os.path.join(outputs_dir, f"{uuid.uuid4().hex}{ext}")
    cv2.imwrite(output_path, output)

    return {
        "output_path":   output_path,
        "faces_found":   counts["faces"],
        "plates_found":  counts["plates"],
        "screens_found": counts["screens"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point — video
# ──────────────────────────────────────────────────────────────────────────────

def process_video(
    input_path: str,
    outputs_dir: str,
    models_dir: str,
    max_duration_sec: float = 12.0,
    max_width: int = 960,
) -> dict:
    """
    Context-aware privacy filtering pipeline for a short video clip.

    Reads the clip frame-by-frame, runs the same detect → filter pipeline used
    for images on every frame (sharing one ThreadPoolExecutor across frames),
    and re-encodes the result as H.264/mp4v MP4 (no audio track — OpenCV's
    VideoCapture/VideoWriter is video-only).

    `max_duration_sec` caps processing time and output size for the demo app —
    frames beyond the cap are dropped and `truncated` is reported True.
    `max_width` downsizes very large frames before detection for speed.

    Returns dict:
        output_path      : str
        faces_found      : int  (summed across all processed frames)
        plates_found     : int
        screens_found    : int
        frames_processed : int
        truncated        : bool
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not read video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_available = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    max_frames = int(max_duration_sec * fps) if max_duration_sec else total_available

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_width / src_w) if src_w > max_width else 1.0
    out_w, out_h = max(1, int(src_w * scale)), max(1, int(src_h * scale))

    os.makedirs(outputs_dir, exist_ok=True)
    output_path = os.path.join(outputs_dir, f"{uuid.uuid4().hex}.mp4")
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h)
    )
    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not open video writer for output.")

    totals = {"faces": 0, "plates": 0, "screens": 0}
    frame_idx = 0
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            while frame_idx < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if scale != 1.0:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                annotated, counts = process_frame(frame, models_dir, executor=pool)
                writer.write(annotated)
                for k in totals:
                    totals[k] += counts[k]
                frame_idx += 1
    finally:
        cap.release()
        writer.release()

    if frame_idx == 0:
        try: os.remove(output_path)
        except OSError: pass
        raise ValueError("No readable frames found in video.")

    return {
        "output_path":      output_path,
        "faces_found":      totals["faces"],
        "plates_found":     totals["plates"],
        "screens_found":    totals["screens"],
        "frames_processed": frame_idx,
        "truncated":        frame_idx >= max_frames and (total_available == 0 or frame_idx < total_available),
    }
