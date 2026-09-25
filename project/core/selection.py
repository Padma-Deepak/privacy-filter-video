"""Selection validation. IDs are class-qualified and refer to one analysis only."""
import math


def match_face_selection(box: list, frame: list) -> str:
    """Attach a drawn face rectangle to one unambiguous raw detection.

    Never treat the rectangle as a stationary hole in the redaction mask and
    never exempt plates/screens. Missing or ambiguous matches fail closed.
    """
    from core.tracking import iou
    x, y, w, h = box
    candidates = []
    for record in frame:
        if record["class"] != "faces" or record["source"] != "detected":
            continue
        raw = record.get("raw_box") or record["box"]
        rx, ry, rw, rh = raw
        if not (x <= rx + rw / 2 <= x + w and y <= ry + rh / 2 <= y + h):
            continue
        score = iou(box, raw)
        if score >= 0.2:
            candidates.append((score, record["id"]))
    candidates.sort(reverse=True)
    if not candidates:
        raise ValueError("No face track matched that box. Draw closely around the face, or pause on a clearer frame.")
    if len(candidates) > 1 and candidates[1][0] >= candidates[0][0] * 0.75:
        raise ValueError("That box covers more than one possible face. Draw a tighter box around one person.")
    return candidates[0][1]


def validate_choices(value: dict, tracks: dict, frame_count: int) -> dict:
    """Missing choices hide by default; reject typos instead of revealing tracks."""
    if not isinstance(value, dict) or any(key not in tracks for key in value):
        raise ValueError("Choices contain an unknown track")
    result = {}
    for key, choice in value.items():
        if not isinstance(choice, dict) or choice.get("mode") not in {"hide", "keep", "range"}:
            raise ValueError("Choose hide, keep or range")
        if choice["mode"] == "range":
            start, end = choice.get("start"), choice.get("end")
            if type(start) is not int or type(end) is not int or not 0 <= start <= end < frame_count:
                raise ValueError("Frame range must be within the clip (zero-based, inclusive)")
            result[key] = {"mode": "range", "start": start, "end": end}
        else:
            result[key] = {"mode": choice["mode"]}
    return result


def is_hidden(key: str, frame: int, choices: dict) -> bool:
    choice = choices.get(key, {"mode": "hide"})
    if choice["mode"] == "keep":
        return False
    return choice["mode"] == "hide" or choice["start"] <= frame <= choice["end"]


def validate_manual(value: list, width: int, height: int, frame_count: int) -> list:
    """Manual rectangles are fixed in space over an explicit inclusive range."""
    if not isinstance(value, list) or len(value) > 100:
        raise ValueError("At most 100 manual regions are allowed")
    result = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Invalid manual region")
        box = item.get("box")
        if not isinstance(box, list) or len(box) != 4 or any(
            type(n) not in (int, float) or not math.isfinite(n) for n in box
        ):
            raise ValueError("Manual box must have four finite coordinates")
        x, y, w, h = map(int, box)
        if not (0 <= x < width and 0 <= y < height and w > 0 and h > 0
                and x + w <= width and y + h <= height):
            raise ValueError("Manual box must fit inside the image")
        start, end = item.get("start"), item.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start <= end < frame_count:
            raise ValueError("Manual frame range must fit inside the clip")
        result.append({"box": [x, y, w, h], "start": start, "end": end})
    return result
