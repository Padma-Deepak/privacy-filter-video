# Privacy Filter (Video)

A locally-run Python web app that automatically detects and anonymizes sensitive visual information in **images and short videos** — no cloud services, no API keys, nothing ever leaves your machine.

| Detected Region | Filter Applied |
|---|---|
| Human faces | Gaussian blur |
| License plates | Black mask |
| Screens, phones, laptops | Pixelation / mosaic |

Detection uses OpenCV Haar Cascades, an optional DNN SSD face detector, and YOLOv8 (`ultralytics`). Video is processed frame-by-frame through the same pipeline and re-encoded as MP4.

The app itself lives in [`project/`](project/) — see **[project/README.md](project/README.md)** for installation, usage, and known limitations.

## Quick start

```bash
cd project
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.

## Repo layout

```
.
├── project/            ← the Flask app (see project/README.md)
├── images_CV_AAT/       ← sample test images
└── instructions.txt     ← original PRD / build spec
```


