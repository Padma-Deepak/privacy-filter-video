"""Measure YuNet on the identical WIDER subset, original image sizes, IoU 0.5."""
import csv
from run_baseline import _load_subset, _load_images, _eval_face_detector, PROJECT_DIR, RESULTS_DIR
from core.faces import detect_faces

if __name__ == '__main__':
    images = _load_images(_load_subset())
    rows = []
    for threshold in (0.6, 0.8):
        row = _eval_face_detector(f'yunet-{threshold}', lambda img, gray: detect_faces(img, PROJECT_DIR / 'models', threshold), images)
        row['notes'] = 'OpenCV YuNet 2023mar; full image resolution; detection only, not video export FPS'
        rows.append(row)
        print(row, flush=True)
    with (RESULTS_DIR / 'yunet_local_arm64.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
