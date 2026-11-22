# main.py
import os
import cv2
from datetime import datetime, timedelta

from run_detection import detect_faces
from face_recognizer import FaceRecognizer
from attendance_tracker import AttendanceTracker
from attentiveness_detector import AttentivenessDetector

# -----------------------------
# Folders
# -----------------------------
TRAINING_FOLDER = "training_images"   # training_images/<student>/*.jpg
GROUP_FOLDER = "face_detection"       # snap1.jpg, snap2.jpg ...
OUTPUT_FOLDER = "cropped_faces"
REPORT_FOLDER = "reports"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# -----------------------------
# Settings
# -----------------------------
DB_PATH = "face_database.pkl"

# ✅ FIX: Raised from 0.45 → 0.55. At 0.45 most real classroom photos
#    don't match even the correct person. 0.55 is a safe starting point;
#    raise to 0.60 if you still get misses, lower to 0.50 if wrong people match.
TOLERANCE = 0.55

CLASS_START_TIME = "09:00"
LATE_THRESHOLD_MIN = 10
SNAPSHOT_INTERVAL_SECONDS = 60

# -----------------------------
# Initialize Recognizer (load DB)
# -----------------------------
recognizer = FaceRecognizer(tolerance=TOLERANCE)
if os.path.exists(DB_PATH):
    recognizer.load_database(DB_PATH)
    recognizer.debug_database()   # ✅ always show what's in the DB at startup
else:
    raise FileNotFoundError(
        f"❌ {DB_PATH} not found. First run: python register_faces.py "
        f"to build the database from {TRAINING_FOLDER}/<person> folders."
    )

# -----------------------------
# Initialize modules
# -----------------------------
attendance_tracker = AttendanceTracker(
    class_start_time=CLASS_START_TIME,
    late_threshold_minutes=LATE_THRESHOLD_MIN
)
attentiveness_detector = AttentivenessDetector()

recognized_faces_by_time = {}   # {timestamp_key: [{name, confidence}, ...]}
detected_faces_by_time = {}     # {timestamp_key: [{face_image}, ...]}

# enrolled students from folder names (authoritative list)
enrolled_students = sorted([
    d for d in os.listdir(TRAINING_FOLDER)
    if os.path.isdir(os.path.join(TRAINING_FOLDER, d))
])

print("\n=== Face Recognition Pipeline Started ===")
print(f"Enrolled students ({len(enrolled_students)}): {enrolled_students}")
print(f"Tolerance: {TOLERANCE}")

# -----------------------------
# Step 1: Process snapshots
# -----------------------------
print("\n[Step 1] Processing snapshots...")

snapshot_files = sorted([
    f for f in os.listdir(GROUP_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

if not snapshot_files:
    raise FileNotFoundError(f"❌ No images found in '{GROUP_FOLDER}/'")

total_faces_saved = 0
total_snapshots = 0

base_time = datetime.strptime(CLASS_START_TIME + ":00", "%H:%M:%S")

for idx, img_name in enumerate(snapshot_files, start=1):
    img_path = os.path.join(GROUP_FOLDER, img_name)
    group_image = cv2.imread(img_path)
    if group_image is None:
        print(f"  ⚠ Cannot read: {img_name}, skipping")
        continue

    total_snapshots += 1

    snap_time = (base_time + timedelta(seconds=(idx - 1) * SNAPSHOT_INTERVAL_SECONDS)).strftime("%H:%M:%S")
    timestamp_key = f"{snap_time}_{idx}"

    faces, boxes = detect_faces(group_image, save_cropped=True, snapshot_id=f"snap{idx:03d}")
    total_faces_saved += len(faces)
    print(f"\n  [{img_name}] → {len(faces)} face(s) detected")

    recognized_faces_by_time[timestamp_key] = []
    detected_faces_by_time[timestamp_key] = []

    for i, (face, box) in enumerate(zip(faces, boxes), start=1):
        # ✅ FIX: face is a BGR numpy array from OpenCV — recognize_face now
        #    handles the BGR→RGB conversion explicitly and correctly.
        name, dist = recognizer.recognize_face(face)

        if name is None:
            display_name = "Unknown"
            normalized_name = "Unknown"
            print(f"    Face {i}: ❓ Unknown (no match within tolerance {TOLERANCE})")
        else:
            display_name = name
            normalized_name = name
            print(f"    Face {i}: ✅ {name} (distance={dist:.3f})")

        recognized_faces_by_time[timestamp_key].append({
            "name": normalized_name,
            "confidence": dist
        })

        detected_faces_by_time[timestamp_key].append({
            "face_image": face
        })

# -----------------------------
# Step 2: Attendance + Attentiveness
# -----------------------------
print("\n[Step 2] Attendance + Attentiveness Summary...")

attendance_records = attendance_tracker.track_attendance(
    recognized_faces=recognized_faces_by_time,
    total_snapshots=total_snapshots
)

attendance_records = attendance_tracker.mark_absent_students(
    attendance_records=attendance_records,
    enrolled_students=enrolled_students
)

presence_percent = attendance_tracker.calculate_presence_percentage(attendance_records)

attentiveness_stats = attentiveness_detector.batch_analyze_attentiveness(
    recognized_faces=recognized_faces_by_time,
    detected_faces=detected_faces_by_time
)

# ✅ Print attendance summary to console for easy debugging
print("\n--- Attendance Summary ---")
for student in sorted(enrolled_students):
    rec = attendance_records.get(student, {})
    status = rec.get("status", "Absent")
    pct = presence_percent.get(student, 0)
    first = rec.get("first_detected", "—")
    late = " (LATE)" if rec.get("is_late") else ""
    print(f"  {student}: {status}{late} | first seen: {first} | presence: {pct}%")

# -----------------------------
# Export combined report
# -----------------------------
try:
    import pandas as pd

    rows = []
    for student in sorted(set(enrolled_students)):
        rec = attendance_records.get(student, {})
        att = attentiveness_stats.get(student, {})

        status = rec.get("status", "Absent")
        late = rec.get("is_late", False)

        total = att.get("total_snapshots", 0)
        attentive = att.get("attentive_count", 0)
        inattentive = att.get("inattentive_count", 0)

        att_percent = round((attentive / total) * 100, 2) if total else 0.0

        rows.append({
            "name": student,
            "status": status,
            "late": late,
            "first_detected": rec.get("first_detected"),
            "present_count": rec.get("present_count", 0),
            "total_snapshots": rec.get("total_snapshots", total_snapshots),
            "presence_percent": presence_percent.get(student, 0),
            "attentive_count": attentive,
            "inattentive_count": inattentive,
            "attentiveness_percent": att_percent
        })

    out_path = os.path.join(REPORT_FOLDER, "class_report.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n✅ Saved combined report: {out_path}")

except Exception as e:
    print(f"(Skipped combined CSV export) Reason: {e}")

print("\n=== Face Recognition Pipeline Finished ===")
print(f"Total snapshots processed: {total_snapshots}")
print(f"Total faces saved/cropped: {total_faces_saved}")
