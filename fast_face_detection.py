from typing import List, Tuple
import numpy as np
import cv2
import face_recognition
from utils import extract_face


class FastFaceDetection:
    """
    Face detection using face_recognition's built-in detector.
    - model="hog" : fast on CPU
    - model="cnn" : more accurate (slow on CPU, fast on GPU)
    Returns:
      faces: list of cropped face images (np.ndarray)
      boxes: list of [x1, y1, x2, y2]
    """

    def __init__(
        self,
        model: str = "hog",          # "hog" (fast) or "cnn" (accurate)
        upsample_times: int = 1,     # increase to 2 to detect smaller faces (slower)
        padding_ratio: float = 0.15  # crop padding for better recognition
    ):
        assert model in ("hog", "cnn")
        self.model = model
        self.upsample_times = upsample_times
        self.padding_ratio = padding_ratio

    def __call__(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[List[int]]]:
        faces: List[np.ndarray] = []
        boxes: List[List[int]] = []

        if image is None or image.size == 0:
            return faces, boxes

        h, w = image.shape[:2]

        # face_recognition expects RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # returns [(top, right, bottom, left), ...]
        locs = face_recognition.face_locations(
            rgb,
            number_of_times_to_upsample=self.upsample_times,
            model=self.model
        )

        for (top, right, bottom, left) in locs:
            x1, y1, x2, y2 = left, top, right, bottom

            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            pad_w = int(bw * self.padding_ratio)
            pad_h = int(bh * self.padding_ratio)

            x1p = max(0, x1 - pad_w)
            y1p = max(0, y1 - pad_h)
            x2p = min(w, x2 + pad_w)
            y2p = min(h, y2 + pad_h)

            bbox = [x1p, y1p, x2p, y2p]
            face = extract_face(image, bbox)

            if face is not None:
                faces.append(face)
                boxes.append(bbox)

        return faces, boxes
