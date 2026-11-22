# register_faces.py
# Run this ONCE (or whenever you add new students) to build face_database.pkl
from face_recognizer import FaceRecognizer
import os

KNOWN_DIR = "training_images"
DB_PATH = "face_database.pkl"

# ✅ Use the same tolerance you'll use in main.py
recognizer = FaceRecognizer(tolerance=0.55)

training_data = {}
for person in sorted(os.listdir(KNOWN_DIR)):
    person_path = os.path.join(KNOWN_DIR, person)
    if os.path.isdir(person_path):
        images = [
            os.path.join(person_path, img)
            for img in sorted(os.listdir(person_path))
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if images:
            training_data[person] = images
            print(f"  Found {len(images)} image(s) for '{person}'")
        else:
            print(f"  ⚠ No images found for '{person}' — skipping")

if not training_data:
    raise ValueError(f"❌ No training data found in '{KNOWN_DIR}/'. "
                     f"Structure should be: {KNOWN_DIR}/<student_name>/*.jpg")

print(f"\nRegistering {len(training_data)} student(s)...")
recognizer.register_faces(training_data, add=False)
recognizer.save_database(DB_PATH)
recognizer.debug_database()
print(f"\n✅ Done. Now run: python main.py")
