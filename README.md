**Smart Face Attendance System**

An Intelligent Face Recognition Framework for Attentiveness Analysis and Latecomer Detection in Smart Classrooms

**Overview**

The Smart Face Attendance System is an intelligent classroom monitoring solution designed to automate student attendance tracking, identify latecomers, and analyze classroom attentiveness using computer vision and facial recognition techniques.

The system uses facial detection and embedding-based facial recognition to identify registered students from classroom images. It also incorporates facial landmark analysis to monitor attentiveness based on Eye Aspect Ratio (EAR).

The project aims to reduce manual attendance efforts while providing additional insights into classroom engagement.

**Key Features**

- Automated Face Detection: Detects multiple faces in classroom images using MediaPipe.
- Face Recognition: Identifies registered students using embedding-based facial recognition.
- Automated Attendance: Records student attendance based on successful face recognition.
- Latecomer Detection: Identifies students arriving after the defined attendance deadline.
- Attentiveness Analysis: Uses Eye Aspect Ratio (EAR) to detect eye closure and estimate attentiveness.
- Attendance Reports: Generates structured attendance reports in CSV format.

**Tech Stack**

| Component | Technology |
|---|---|
| Programming Language | Python 3.10.8 |
| Face Detection | MediaPipe |
| Facial Recognition | Embedding-based Face Recognition |
| Facial Landmark Detection | MediaPipe Face Mesh |
| Attentiveness Analysis | Eye Aspect Ratio (EAR) |
| Data Processing | NumPy, OpenCV |
| Report Generation | Pandas |
| Output | CSV Reports |

**System Workflow**

1. **Face Registration:** Capture and register student facial images to create facial embeddings.
2. **Face Detection:** Process classroom images and detect individual faces.
3. **Face Recognition:** Generate facial embeddings and compare them against registered student embeddings.
4. **Attendance Tracking:** Mark recognized students as present.
5. **Latecomer Detection:** Compare attendance timestamps against the configured attendance deadline.
6. **Attentiveness Analysis:** Analyze eye landmarks using EAR to identify prolonged eye closure.
7. **Report Generation:** Export attendance information into structured CSV reports.

## 📊 System Configuration

The current implementation uses the following parameters:

| Parameter | Configuration |
|---|---|
| Attendance Start Time | 09:00 AM |
| Late Arrival Threshold | 09:10 AM |
| EAR Threshold | 0.25 |
| Consecutive Frames for Eye Closure | 2 |
| Face Recognition Tolerance | 0.45 (default) |

**Project Structure**

```text
SmartAttendance/
│
├── main.py
├── run_detection.py
├── face_recognizer.py
├── fast_face_detection.py
├── register_faces.py
├── attendance_tracker.py
├── attentiveness_detector.py
├── utils.py
│
├── reports/
│   └── class_report.csv
│
└── README.md
```

## 📈 Performance

The system was evaluated through classroom image processing and facial recognition experiments.

| Evaluation Metric | Reported Result |
|---|---:|
| Face Recognition Accuracy | ~96% |
| Attentiveness Detection Accuracy | 84.7% |

*Note: These are results from the project's evaluation and may vary depending on image quality, lighting conditions, face visibility, and classroom environment.*

**Getting Started**

### Prerequisites

- Python 3.10.8
- pip package manager
- Classroom images for testing

### Installation

**1. Clone the repository**

```bash
git clone <YOUR_REPOSITORY_URL>
```

**2. Navigate to the project directory**

```bash
cd SmartAttendance
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the application**

```bash
python main.py
```

*Note: The dependency list and execution instructions should be aligned with the final repository files.*

## 🔮 Future Enhancements

The project is planned to evolve into a full-stack classroom attendance and analytics platform.

Potential enhancements include:

- React-based interactive frontend.
- Backend API development.
- Database integration for student profiles and attendance records.
- Real-time attendance monitoring.
- Improved multi-face detection in crowded classroom environments.
- Interactive attendance and attentiveness dashboards.
- Secure authentication and role-based access.

## 🎓 Research Background

This project is based on the research work titled:

**"An Intelligent Face Recognition Framework for Attentiveness Analysis and Latecomer Detection in Smart Classrooms."**

The research explores the integration of facial recognition, automated attendance tracking, and computer vision-based attentiveness analysis to improve classroom monitoring.

## 👩‍💻 Author

**Anshika Singh**


---

⭐ If you find this project interesting, feel free to explore the repository!
