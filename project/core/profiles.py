"""Validated local redaction presets; detector quality is not implied by a preset."""
from pathlib import Path
import math

import yaml

CLASSES = {"faces", "plates", "screens"}
PROFILE_DIR = Path(__file__).resolve().parents[1] / "profiles"


def load_profile(name: str = "creator", custom: str | None = None) -> dict:
    """Read a bundled preset or a bounded YAML document (never a user path)."""
    if name not in {"creator", "journalist", "custom"}:
        raise ValueError("Unknown profile")
    if name == "custom":
        if not isinstance(custom, str) or not 0 < len(custom) <= 16000:
            raise ValueError("Custom profile must contain at most 16000 characters")
        text = custom
    else:
        text = (PROFILE_DIR / f"{name}.yaml").read_text()
    try:
        profile = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError("Invalid profile YAML") from exc
    required = {"name", "classes", "filters", "padding", "blur_kernel", "blur_sigma",
                "pixel_size", "track_buffers", "audio", "strip_metadata", "report",
                "detector", "face_confidence", "screen_confidence"}
    if not isinstance(profile, dict) or set(profile) != required:
        raise ValueError("Profile fields must match the bundled YAML schema")
    if not isinstance(profile["name"], str) or not 1 <= len(profile["name"]) <= 80:
        raise ValueError("Profile name must contain 1–80 characters")
    classes = profile["classes"]
    if not isinstance(classes, list) or not classes or any(c not in CLASSES for c in classes):
        raise ValueError("Classes must be faces, plates or screens")
    for field in ("filters", "track_buffers"):
        if not isinstance(profile[field], dict) or set(profile[field]) != CLASSES:
            raise ValueError(f"{field} must specify faces, plates and screens")
    if any(v not in {"blur", "solid", "pixelate"} for v in profile["filters"].values()):
        raise ValueError("Filter must be blur, solid or pixelate")
    for field, low, high in (("padding", 0, 1), ("blur_sigma", 10, 100),
                              ("face_confidence", 0.01, 1), ("screen_confidence", 0.01, 1)):
        value = profile[field]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{field} must be between {low} and {high}")
    for field, low, high in (("blur_kernel", 31, 301), ("pixel_size", 8, 100)):
        if type(profile[field]) is not int or not low <= profile[field] <= high:
            raise ValueError(f"{field} must be an integer between {low} and {high}")
    if profile["blur_kernel"] % 2 != 1:
        raise ValueError("blur_kernel must be odd")
    if any(type(v) is not int or not 0 <= v <= 120 for v in profile["track_buffers"].values()):
        raise ValueError("Track buffers must be integers between 0 and 120")
    if profile["audio"] not in {"keep", "mute"} or profile["detector"] != "yunet":
        raise ValueError("Audio must be keep/mute; currently supported detector is yunet")
    if any(type(profile[f]) is not bool for f in ("strip_metadata", "report")):
        raise ValueError("strip_metadata and report must be booleans")
    if not profile["strip_metadata"]:
        raise ValueError("Metadata removal must remain enabled")
    return profile


def redact(frame, box, cls: str, profile: dict) -> None:
    """Apply the chosen filter in-place; journalist uses exact solid pixels."""
    import detector
    kind = profile["filters"][cls]
    if kind == "solid":
        detector.apply_black_mask(frame, *box)
    elif kind == "blur":
        kernel = profile["blur_kernel"]
        detector.apply_gaussian_blur(frame, *box, kernel=(kernel, kernel), sigma=profile["blur_sigma"])
    else:
        detector.apply_pixelation(frame, *box, blocks=profile["pixel_size"])
