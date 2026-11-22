import cv2

def extract_face(image, bbox):
    x1, y1, x2, y2 = bbox

    face = image[y1:y2, x1:x2]

    if face.size == 0:
        return None

    return face
