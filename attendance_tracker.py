# attendance_tracker.py
from datetime import datetime, timedelta
from collections import defaultdict

class AttendanceTracker:
    def __init__(self, class_start_time="09:00", late_threshold_minutes=10):
        """
        class_start_time: "HH:MM" (24-hour format) e.g. "09:00"
        late_threshold_minutes: minutes after class_start_time to mark late (e.g. 10 => after 09:10 late)
        """
        self.class_start_time = class_start_time
        self.late_threshold_minutes = late_threshold_minutes

    def parse_timestamp(self, timestamp: str) -> datetime:
        """
        Accepts:
          - "HH:MM:SS"
          - "HH:MM:SS_#"(your main.py uses this for uniqueness)
        Returns a datetime object (date part is dummy, time part is used).
        """
        # ✅ FIX: handle keys like "09:02:15_3"
        if "_" in timestamp:
            timestamp = timestamp.split("_")[0]

        try:
            return datetime.strptime(timestamp, "%H:%M:%S")
        except ValueError:
            # allow "HH:MM" too (fallback)
            try:
                return datetime.strptime(timestamp, "%H:%M")
            except ValueError as e:
                raise ValueError(f"Could not parse timestamp: {timestamp}") from e

    def _class_start_dt(self) -> datetime:
        # class_start_time is HH:MM
        return datetime.strptime(self.class_start_time, "%H:%M")

    def _late_cutoff_dt(self) -> datetime:
        return self._class_start_dt() + timedelta(minutes=self.late_threshold_minutes)

    def track_attendance(self, recognized_faces: dict, total_snapshots: int):
        """
        recognized_faces: {timestamp_key: [{"name": str, "confidence": float}, ...], ...}
        total_snapshots: total group images processed
        Returns:
          attendance_records: dict
            {
              "student": {
                  "first_detected": "HH:MM:SS" or None,
                  "present_count": int,
                  "total_snapshots": int,
                  "is_late": bool,
                  "status": "Present"/"Absent" (optional)
              }
            }
        """
        # Initialize structure
        attendance_records = defaultdict(lambda: {
            "first_detected": None,
            "present_count": 0,
            "total_snapshots": total_snapshots,
            "is_late": False
        })

        # Sort timestamps in chronological order (parse-safe)
        sorted_times = sorted(recognized_faces.keys(), key=lambda t: self.parse_timestamp(t))

        late_cutoff = self._late_cutoff_dt()

        for t_key in sorted_times:
            seen_list = recognized_faces.get(t_key, [])

            # Normalize timestamp string for storage (keep HH:MM:SS only)
            clean_time = t_key.split("_")[0] if "_" in t_key else t_key

            for item in seen_list:
                name = item.get("name", "Unknown")

                # ignore unknowns for attendance
                if name == "Unknown":
                    continue

                # count presence
                attendance_records[name]["present_count"] += 1

                # set first detected and late
                if attendance_records[name]["first_detected"] is None:
                    attendance_records[name]["first_detected"] = clean_time

                    # late check based on first detection time
                    first_dt = self.parse_timestamp(clean_time)
                    attendance_records[name]["is_late"] = first_dt > late_cutoff

        # Optional: set default status for detected people
        for name, rec in attendance_records.items():
            if rec["first_detected"] is not None:
                rec["status"] = "Present"

        return dict(attendance_records)

    def mark_absent_students(self, attendance_records: dict, enrolled_students: list):
        """
        Ensure every enrolled student appears in report.
        If not detected, mark absent.
        """
        for student in enrolled_students:
            if student not in attendance_records:
                attendance_records[student] = {
                    "first_detected": None,
                    "present_count": 0,
                    "total_snapshots": attendance_records[next(iter(attendance_records))]["total_snapshots"]
                        if attendance_records else 0,
                    "is_late": False,
                    "status": "Absent"
                }
            else:
                # if never detected
                if attendance_records[student].get("first_detected") is None:
                    attendance_records[student]["status"] = "Absent"
                else:
                    attendance_records[student]["status"] = "Present"
        return attendance_records

    def calculate_presence_percentage(self, attendance_records: dict):
        """
        presence% = present_count / total_snapshots * 100
        """
        presence_percent = {}
        for student, rec in attendance_records.items():
            total = rec.get("total_snapshots", 0) or 0
            present = rec.get("present_count", 0) or 0
            pct = round((present / total) * 100, 2) if total else 0.0
            presence_percent[student] = pct
        return presence_percent
