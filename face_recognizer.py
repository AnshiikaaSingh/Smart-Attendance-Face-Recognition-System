# face_recognizer.py
import face_recognition
import pickle
import numpy as np

class FaceRecognizer:
    def __init__(self, tolerance: float = 0.55):
        self.database = {}   # {name: [enc1, enc2, ...]}
        self.tolerance = tolerance

    @staticmethod
    def ensure_rgb(image, is_bgr: bool = True):
        """
        Explicitly convert BGR→RGB only when told to.
        face_recognition always needs RGB input.

        Args:
            image: numpy array
            is_bgr: True  → image came from OpenCV (needs flip)
                    False → image is already RGB (load_image_file / face_recognition detector output)
        """
        if is_bgr and hasattr(image, "shape") and len(image.shape) == 3 and image.shape[2] == 3:
            return image[:, :, ::-1]   # BGR → RGB
        return image

    def register_faces(self, face_dict, add: bool = True):
        """
        face_dict: {name: path_str} or {name: [path1, path2, ...]}
        Paths are loaded via face_recognition.load_image_file → already RGB.
        """
        for name, faces in face_dict.items():
            if not isinstance(faces, (list, tuple)):
                faces = [faces]

            if (name not in self.database) or (not add):
                self.database[name] = []

            added = 0
            for face in faces:
                if isinstance(face, str):
                    # load_image_file returns RGB — no conversion needed
                    image = face_recognition.load_image_file(face)
                    is_bgr = False
                else:
                    # numpy array from OpenCV → assume BGR
                    image = face
                    is_bgr = True

                image_rgb = self.ensure_rgb(image, is_bgr=is_bgr)
                encodings = face_recognition.face_encodings(image_rgb)
                if not encodings:
                    print(f"  ⚠ No face found in image for '{name}', skipping")
                    continue

                self.database[name].append(encodings[0])
                added += 1

            if added == 0:
                print(f"❌ No usable face found for '{name}' — skipping registration")
            else:
                print(f"✅ Registered {added} encoding(s) for '{name}'")

    def recognize_face(self, face_bgr: np.ndarray):
        """
        face_bgr: numpy array crop from OpenCV (BGR).
        Returns (name, distance) or (None, None).
        """
        # Crops from FastFaceDetection / extract_face come from cv2.imread → BGR
        face_rgb = self.ensure_rgb(face_bgr, is_bgr=True)

        encodings = face_recognition.face_encodings(face_rgb)
        if not encodings:
            return None, None

        face_encoding = encodings[0]
        best_distance = 1.0
        best_name = None

        for name, enc_list in self.database.items():
            if not enc_list:
                continue
            distances = face_recognition.face_distance(enc_list, face_encoding)
            min_dist = float(np.min(distances))
            if min_dist < self.tolerance and min_dist < best_distance:
                best_distance = min_dist
                best_name = name

        if best_name is None:
            return None, None
        return best_name, best_distance

    def save_database(self, path="face_database.pkl"):
        obj = {"tolerance": self.tolerance, "database": self.database}
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"✅ Face database saved to '{path}' ({len(self.database)} identities)")

    def load_database(self, path="face_database.pkl"):
        with open(path, "rb") as f:
            obj = pickle.load(f)

        # backward compatibility: old format was {name: single_encoding}
        if isinstance(obj, dict) and "database" in obj:
            self.tolerance = obj.get("tolerance", self.tolerance)
            self.database = obj["database"]
        else:
            self.database = {k: [v] for k, v in obj.items()}

        total_encs = sum(len(v) for v in self.database.values())
        print(f"✅ Loaded database: {len(self.database)} identities, {total_encs} total encodings")

    def debug_database(self):
        """Print a summary of what's in the database."""
        print("\n=== Face Database Debug ===")
        if not self.database:
            print("❌ Database is EMPTY! Run register_faces.py first.")
            return
        for name, encs in self.database.items():
            print(f"  {name}: {len(encs)} encoding(s)")
        print(f"  Tolerance: {self.tolerance}")
        print("===========================\n")
