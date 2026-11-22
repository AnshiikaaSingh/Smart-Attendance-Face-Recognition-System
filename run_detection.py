# run_detection.py
import cv2
import os
from datetime import datetime
from fast_face_detection import FastFaceDetection

IMAGE_FOLDER = "face_detection"
OUTPUT_FOLDER = "cropped_faces"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# updated detector settings are inside FastFaceDetection
detector = FastFaceDetection()
face_count = 0


def detect_faces(
    image,
    save_cropped: bool = True,
    snapshot_id: str | None = None,     # e.g., "snap001"
    session_subdir: str | None = None,  # e.g., "S1"
    min_face_size: int = 35             # was 60 (too strict for back-row faces)
):
    """
    Detects faces in a single image.

    Returns:
        faces: list of cropped face images (BGR)
        boxes: list of bounding boxes (x1, y1, x2, y2) - SAME coords as image shown/saved
    """
    global face_count

    faces, boxes = detector(image)

    # Filter out tiny/garbage crops
    filtered_faces, filtered_boxes = [], []
    for face, box in zip(faces, boxes):
        if face is None or getattr(face, "size", 0) == 0:
            continue

        h, w = face.shape[:2]
        if h < min_face_size or w < min_face_size:
            # Debug if you want:
            # print(f"Dropped small face: {w}x{h} at box={box}")
            continue

        filtered_faces.append(face)
        filtered_boxes.append(box)

    faces, boxes = filtered_faces, filtered_boxes

    if save_cropped and faces:
        out_dir = OUTPUT_FOLDER
        if session_subdir:
            out_dir = os.path.join(OUTPUT_FOLDER, session_subdir)
        os.makedirs(out_dir, exist_ok=True)

        for idx, face in enumerate(faces, start=1):
            face_count += 1

            if snapshot_id is not None:
                filename = f"{snapshot_id}_face{idx:02d}.jpg"
            else:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"face_{face_count}_{timestamp}.jpg"

            cv2.imwrite(os.path.join(out_dir, filename), face)

    return faces, boxes


def process_folder(
    folder: str = IMAGE_FOLDER,
    save_cropped: bool = True,
    session_subdir: str | None = None,
    show_gui: bool = True
):
    global face_count
    face_count = 0

    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    for idx, img_name in enumerate(files, start=1):
        img_path = os.path.join(folder, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print("Cannot read:", img_name)
            continue

        snapshot_id = f"snap{idx:03d}_{os.path.splitext(img_name)[0]}"
        faces, boxes = detect_faces(
            image,
            save_cropped=save_cropped,
            snapshot_id=snapshot_id,
            session_subdir=session_subdir
        )

        print(f"{img_name} → Faces detected (after filtering): {len(faces)}")

        if show_gui:
            # Draw boxes for visualization
            vis = image.copy()

            # NOTE: If your original image is VERY wide and FastFaceDetection resized internally,
            # boxes correspond to resized image. For easiest correctness, keep max_width high (1600/2000).
            # If you still see mismatch, tell me your image widths and I’ll add scale-back.
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow("Face Detection", vis)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    if show_gui:
        cv2.destroyAllWindows()

    if save_cropped:
        print(f"Total faces saved: {face_count}")
        print(f"Crops saved under: {OUTPUT_FOLDER}" + (f"/{session_subdir}" if session_subdir else ""))


if __name__ == "__main__":
    print("Program started")
    process_folder(folder=IMAGE_FOLDER, save_cropped=True, session_subdir=None, show_gui=True)
    print("Program finished")
