# Context-Aware Privacy Filtering System

A locally-run Python web application that automatically detects and anonymizes sensitive visual information in uploaded **images and short videos** using **context-appropriate filtering techniques**:

| Detected Region | Filter Applied |
|---|---|
| Human faces | Gaussian blur |
| License plates | Black mask (solid block) |
| Screens, phones, laptops | Pixelation / mosaic |

Detection uses OpenCV Haar Cascades, DNN SSD, and YOLOv8. No image or video data is ever sent to an external server — the entire pipeline runs on your machine.

---

## Installation

```bash
pip install -r requirements.txt
```

This installs: `flask`, `opencv-python`, `numpy`, `ultralytics` (YOLOv8).

> **YOLOv8 model:** `yolov8n.pt` (~6 MB) is downloaded automatically from Ultralytics on the first run. An internet connection is required for that first run only.

---

## Optional: DNN Face Model (Improves Face Detection Accuracy)

Download these two files and place them in the `models/` folder:

1. **`res10_300x300_ssd_iter_140000.caffemodel`**
   https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

2. **`deploy.prototxt`**
   https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

```
project/models/
├── res10_300x300_ssd_iter_140000.caffemodel
└── deploy.prototxt
```

> The Haar Cascade XML files (`haarcascade_frontalface_default.xml`, `haarcascade_russian_plate_number.xml`) do **not** need to be downloaded — they are bundled with `opencv-python`.

---

## Running the App

```bash
python app.py
```

Open: `http://127.0.0.1:5000`

Upload a JPEG, PNG, BMP, or WEBP **image**, or an MP4, MOV, AVI, MKV, or WEBM **video**. The system will:
1. Detect faces → apply Gaussian blur
2. Detect license plates → apply black mask
3. Detect screens / phones / laptops → apply pixelation

For video, every frame runs through the same detect → filter pipeline used for images, and the result is re-encoded as an MP4.

The processed file is shown side-by-side with the original and can be downloaded from the results page. Both the upload and output are deleted from disk immediately after the response is sent.

### Video notes

- Clips are capped at **~12 seconds** of processing (later frames are dropped) to keep processing time and page size reasonable for a demo app — this is configurable via `max_duration_sec` in `detector.process_video()`.
- Frames wider than 960px are downscaled before detection for speed.
- Output video has **no audio track** — OpenCV's `VideoCapture`/`VideoWriter` are video-only.

---

## Project Structure

```
project/
├── app.py             ← Flask web server
├── detector.py        ← Detection + context-aware filter pipeline
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── uploads/           ← Temp (auto-cleared)
├── outputs/           ← Temp (auto-cleared)
└── models/            ← Place optional DNN model files here
```

---

## Team

- Rithvik Kumar R K — 1BM23CS269  
- Padma Deepak — 1BM23CS222  
- Sarthaka Mitra GB — 1BM23CS305  
- Hrishikesh R Prasad — 1BM23CS367  

**Course:** Computer Vision | **Instructor:** Dr. A. Sarkunavathi

---

## Known Limitations

- Haar Cascade misses faces turned > 45° from frontal or heavily occluded
- YOLOv8 screen detection requires reasonable object size and clarity
- License plate detection optimized for rectangular formats; non-standard layouts may be missed
- False positives possible on geometric patterns resembling license plates
- Video processing is capped at ~12 seconds and drops audio (see [Video notes](#video-notes))
- Detection runs independently per video frame — there is no temporal tracking, so a detection can flicker on/off between adjacent frames
