import cv2
import numpy as np

def extract_visual_features(image_path):
    """
    Extract basic visual structure from image
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = {
        "num_contours": len(contours),
        "has_edges": bool(np.sum(edges) > 0)
    }

    return features