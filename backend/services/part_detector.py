import cv2


def map_to_expected_parts(detected_parts, concept):
    """
    Map generic detected parts → concept-specific parts
    """

    mapping = {
        "cat": {
            "body": ["tail"],
            "head": ["head", "eyes", "ears"],
            "leg": ["legs"]
        },
        "dog": {
            "body": ["tail"],
            "head": ["head", "eyes", "ears"],
            "leg": ["legs"]
        },
        "tree": {
            "trunk": ["trunk"],
            "branches": ["branches"],
            "leaves": ["leaves"]
        },
        "car": {
            "body": ["body"],
            "wheel": ["wheels"]
        }
    }

    final_parts = []

    if concept in mapping:
        for part in detected_parts:
            if part in mapping[concept]:
                final_parts.extend(mapping[concept][part])

    return list(set(final_parts))


def detect_parts_from_image(image_path: str, concept: str):
    """
    Concept-aware detection + mapping
    """

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    raw_detected = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 100:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        # 🔥 Raw detection (generic)
        if aspect_ratio > 1.5:
            raw_detected.append("body")
        elif aspect_ratio < 0.5:
            raw_detected.append("leg")
        else:
            raw_detected.append("head")

    raw_detected = list(set(raw_detected))

    # 🔥 Map to concept-specific parts
    final_detected = map_to_expected_parts(raw_detected, concept)

    return final_detected