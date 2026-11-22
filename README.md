# Smart Face Attendance System

### An Intelligent Face Recognition Framework for Attentiveness Analysis and Latecomer Detection in Smart Classrooms

## 📌 Overview

The **Smart Face Attendance System** is an intelligent classroom monitoring solution designed to automate student attendance tracking, identify latecomers, and analyze classroom attentiveness using computer vision and facial recognition techniques.

The system uses face detection and embedding-based facial recognition to identify registered students from classroom group photographs. It also incorporates facial landmark analysis to monitor attentiveness based on the Eye Aspect Ratio (EAR).

The project aims to reduce manual attendance efforts while providing additional insights into classroom engagement.

## ✨ Key Features

- **Automated Face Detection:** Detects multiple faces in classroom group photographs using MediaPipe.
- **Face Recognition:** Identifies registered students using embedding-based facial recognition.
- **Automated Attendance:** Records attendance based on successful face recognition.
- **Latecomer Detection:** Identifies students arriving after the predefined attendance deadline.
- **Attentiveness Analysis:** Uses Eye Aspect Ratio (EAR) to detect eye closure and estimate attentiveness.
- **Attendance Reports:** Generates structured attendance reports in CSV format.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.10.8 |
| Face Detection | MediaPipe |
| Facial Recognition | Embedding-based Face Recognition |
| Facial Landmark Detection | MediaPipe Face Mesh |
| Attentiveness Analysis | Eye Aspect Ratio (EAR) |
| Image Processing | OpenCV |
| Data Processing | NumPy, Pandas |
| Report Generation | CSV |

## 📂 Project Structure

```text
SmartAttendance/
│
├── Training_Images/
│   ├── Student_1/
│   ├── Student_2/
│   └── ...
│
├── Classroom_Snapshots/
│   ├── classroom_01.jpg
│   ├── classroom_02.jpg
│   └── ...
│
├── reports/
│
├── main.py
├── run_detection.py
├── face_recognizer.py
├── attendance_tracker.py
├── attentiveness_detector.py
├── register_faces.py
├── utils.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Folder Description

| Folder | Purpose |
|---|---|
| `Training_Images/` | Contains individual student facial images used for student registration and facial embedding generation. |
| `Classroom_Snapshots/` | Contains classroom group photographs used as input for face detection, recognition, and attendance processing. |
| `reports/` | Stores generated attendance reports and related output files. |

**Note:** Student photographs and classroom snapshots are used as input datasets. For privacy reasons, actual student images should not be uploaded to a public repository.

## ⚙️ System Workflow

The system follows a structured workflow consisting of student registration, face recognition, attendance tracking, and attentiveness analysis.

1. **Student Registration:** Individual student images are stored in the `Training_Images/` directory and processed to generate facial embeddings.
2. **Classroom Image Input:** Group photographs captured in the classroom are provided through the `Classroom_Snapshots/` directory.
3. **Face Detection:** MediaPipe detects individual faces present in the classroom photographs.
4. **Face Recognition:** Facial embeddings are generated and compared against registered student embeddings to identify individuals.
5. **Attendance Tracking:** Recognized students are marked present in the attendance records.
6. **Latecomer Detection:** Attendance timestamps are evaluated against the configured attendance deadline to identify late arrivals.
7. **Attentiveness Analysis:** MediaPipe Face Mesh extracts facial landmarks, and EAR is calculated to detect eye closure.
8. **Report Generation:** Attendance information is compiled and exported into structured CSV reports.

## 📊 System Configuration

The current implementation uses the following parameters:

| Parameter | Configuration |
|---|---|
| Attendance Start Time | 09:00 AM |
| Late Arrival Threshold | 09:10 AM |
| EAR Threshold | 0.25 |
| Consecutive Frames for Eye Closure | 2 |
| Face Recognition Tolerance | 0.45 (default) |

## 📈 Performance Evaluation

The system was evaluated through classroom image processing and facial recognition experiments.

| Evaluation Metric | Reported Result |
|---|---:|
| Face Recognition Accuracy | ~96% |
| Attentiveness Detection Accuracy | 84.7% |

*Note: Performance may vary depending on lighting conditions, image quality, face visibility, camera angle, and classroom environment.*

## 🚀 Getting Started

### Prerequisites

- Python 3.10.8
- pip package manager
- Classroom group photographs for testing

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

**4. Add the input images**

- Place individual student photographs inside `Training_Images/`.
- Place classroom group photographs inside `Classroom_Snapshots/`.

**5. Run the application**

```bash
python main.py
```

*Note: Ensure that the required image datasets and configurations are available before executing the application. The execution command and dependency list should match the final implementation.*

## 🔮 Future Enhancements

The project can be further extended into a comprehensive classroom attendance and analytics platform.

Planned enhancements include:

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
