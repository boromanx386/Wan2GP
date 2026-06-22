from __future__ import annotations

import copy
import json
from typing import Any


def _as_bbox(raw: Any) -> list[float] | None:
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    try:
        return [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])]
    except (TypeError, ValueError):
        return None


def _scale_bbox_to_1000(bbox: list[float]) -> list[int]:
    max_val = max(abs(v) for v in bbox)
    if max_val <= 1.0:
        scaled = [v * 1000.0 for v in bbox]
    elif max_val <= 100.0:
        scaled = [v * 10.0 for v in bbox]
    else:
        scaled = bbox[:]
    y1, x1, y2, x2 = scaled
    y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
    x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
    return [int(round(y1)), int(round(x1)), int(round(y2)), int(round(x2))]


def _swap_yx(bbox: list[int]) -> list[int]:
    y1, x1, y2, x2 = bbox
    return _scale_bbox_to_1000([x1, y1, x2, y2])


def _bbox_score(bboxes: list[list[int]]) -> float:
    if not bboxes:
        return 0.0
    widths = [max(0, b[3] - b[1]) for b in bboxes]
    heights = [max(0, b[2] - b[0]) for b in bboxes]
    avg_w = sum(widths) / len(widths)
    avg_h = sum(heights) / len(heights)
    narrow = sum(1 for w in widths if w < 70)
    tiny_h = sum(1 for h in heights if h < 70)
    return avg_w + avg_h - narrow * 120 - tiny_h * 40


def _ensure_min_span(bbox: list[int], min_span: int = 48) -> list[int]:
    y1, x1, y2, x2 = bbox
    h = y2 - y1
    w = x2 - x1
    if h < min_span:
        cy = (y1 + y2) / 2.0
        y1 = int(max(0, round(cy - min_span / 2)))
        y2 = int(min(1000, round(cy + min_span / 2)))
    if w < min_span:
        cx = (x1 + x2) / 2.0
        x1 = int(max(0, round(cx - min_span / 2)))
        x2 = int(min(1000, round(cx + min_span / 2)))
    return [y1, x1, y2, x2]


_DEFAULT_TEXT_DESC = "clear, legible typography contained within the bounding box"


def _element_content_snippet(element: dict[str, Any]) -> str:
    element_type = str(element.get("type") or "obj").strip().lower()
    if element_type == "text":
        literal = str(element.get("text") or "").strip()
        desc = str(element.get("desc") or "").strip()
        if literal and literal.lower() != "text":
            return literal
        if desc and desc.lower() not in {"", "text", _DEFAULT_TEXT_DESC.lower()}:
            return desc[:120]
        return ""
    desc = str(element.get("desc") or "").strip()
    if desc and desc.lower() != "object":
        return desc[:120]
    return ""


def sync_text_element_fields(doc: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Ensure text elements carry a useful desc when only `text` was edited."""
    notes: list[str] = []
    comp = doc.get("compositional_deconstruction")
    if not isinstance(comp, dict):
        return doc, notes
    elements = comp.get("elements")
    if not isinstance(elements, list):
        return doc, notes

    out = copy.deepcopy(doc)
    changed = False
    for element in out["compositional_deconstruction"]["elements"]:
        if not isinstance(element, dict) or str(element.get("type") or "").lower() != "text":
            continue
        literal = str(element.get("text") or "").strip()
        desc = str(element.get("desc") or "").strip()
        if not literal or literal.lower() == "text":
            continue
        if desc and desc.lower() not in {"", "text", _DEFAULT_TEXT_DESC.lower()}:
            continue
        element["desc"] = (
            f'Readable text reading "{literal}" in clear legible typography within the bounding box.'
        )
        changed = True
    if changed:
        notes.append("filled text element descriptions from literal text")
    return out, notes


def enrich_high_level_description(doc: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Mention layout elements missing from the scene summary so generation follows them."""
    notes: list[str] = []
    if not isinstance(doc, dict):
        return doc, notes

    hld = str(doc.get("high_level_description") or "").strip()
    comp = doc.get("compositional_deconstruction")
    if not isinstance(comp, dict):
        return doc, notes
    elements = comp.get("elements")
    if not isinstance(elements, list):
        return doc, notes

    missing: list[str] = []
    hld_lower = hld.lower()
    for element in elements:
        if not isinstance(element, dict):
            continue
        snippet = _element_content_snippet(element)
        if not snippet:
            continue
        key = snippet[:48].lower()
        if key not in hld_lower:
            missing.append(snippet)

    if not missing:
        return doc, notes

    out = copy.deepcopy(doc)
    prefix = hld.rstrip()
    suffix = " Also include: " + "; ".join(missing[:6]) + "."
    out["high_level_description"] = (prefix + suffix).strip()
    notes.append("extended high_level_description with layout element descriptions")
    return out, notes


def sanitize_ideogram_prompt_doc(
    doc: dict[str, Any],
    *,
    allow_axis_swap: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    """Normalize bbox scale/order for Ideogram JSON captions."""
    notes: list[str] = []
    if not isinstance(doc, dict):
        return doc, notes

    comp = doc.get("compositional_deconstruction")
    if not isinstance(comp, dict):
        return doc, notes

    elements = comp.get("elements")
    if not isinstance(elements, list):
        return doc, notes

    out = copy.deepcopy(doc)
    comp = out["compositional_deconstruction"]
    elements = comp["elements"]

    scaled: list[list[int]] = []
    indices: list[int] = []
    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            continue
        bbox = _as_bbox(element.get("bbox"))
        if bbox is None:
            continue
        scaled.append(_scale_bbox_to_1000(bbox))
        indices.append(index)

    if not scaled:
        return out, notes

    if max(max(b) for b in scaled) <= 100:
        notes.append("scaled bbox values from 0–100 to 0–1000")

    if allow_axis_swap:
        narrow_tall = sum(1 for b in scaled if (b[3] - b[1]) < 80 and (b[2] - b[0]) > 150)
        if narrow_tall >= max(2, len(scaled) * 2 // 3):
            notes.append("corrected bbox axis order (y/x swap)")
            scaled = [_swap_yx(b) for b in scaled]
        else:
            original_score = _bbox_score(scaled)
            swapped = [_swap_yx(b) for b in scaled]
            swapped_score = _bbox_score(swapped)
            if swapped_score > original_score + 80:
                notes.append("corrected bbox axis order (y/x swap)")
                scaled = swapped

        narrow_count = sum(1 for b in scaled if (b[3] - b[1]) < 70)
        if narrow_count >= max(2, len(scaled) // 2):
            notes.append("expanded very narrow bbox widths for layout editing")

    for slot, index in enumerate(indices):
        bbox = _ensure_min_span(scaled[slot])
        elements[index]["bbox"] = bbox

    return out, notes


def sanitize_ideogram_prompt_json(json_text: str) -> tuple[str, str, list[str]]:
    doc = json.loads(json_text)
    if not isinstance(doc, dict):
        raise ValueError("Magic Prompt JSON root must be an object.")
    doc, notes = sanitize_ideogram_prompt_doc(doc, allow_axis_swap=True)
    compact = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
    pretty = json.dumps(doc, indent=2, ensure_ascii=False)
    return compact, pretty, notes


def prepare_prompt_for_generation(json_text: str) -> tuple[str, list[str]]:
    """Prepare user-edited layout JSON for generation without corrupting manual bbox edits."""
    doc = json.loads(json_text)
    if not isinstance(doc, dict):
        raise ValueError("Prompt JSON root must be an object.")

    notes: list[str] = []
    doc, sync_notes = sync_text_element_fields(doc)
    notes.extend(sync_notes)
    doc, hld_notes = enrich_high_level_description(doc)
    notes.extend(hld_notes)
    doc, bbox_notes = sanitize_ideogram_prompt_doc(doc, allow_axis_swap=False)
    notes.extend(bbox_notes)
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":")), notes
