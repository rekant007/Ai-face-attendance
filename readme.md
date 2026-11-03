# 🤖 AI Face Recognition Attendance System

An AI-powered attendance management system built with **Python, OpenCV, MediaPipe, and Supabase**. The system recognizes registered students using facial embeddings, verifies user presence through blink-based liveness detection, and automatically records attendance with date and timestamp.

## ✨ Features

- **Face Detection:** Detects faces using OpenCV YuNet.
- **Face Recognition:** Uses OpenCV SFace to generate and compare facial embeddings.
- **Liveness Detection:** Uses MediaPipe Face Landmarker and Eye Aspect Ratio (EAR) to detect eye blinks.
- **Student Registration:** Registers students using their student ID, name, and facial data.
- **Automated Attendance:** Recognizes students and records their attendance.
- **Duplicate Prevention:** Prevents duplicate attendance entries during a session.
- **Absence Management:** Marks students who were not recognized during the attendance session as absent.
- **Cloud Database:** Stores student information, facial embeddings, landmarks, and attendance records in Supabase.
- **Face Image Storage:** Uploads captured face images to Supabase Storage.
- **Timestamp Tracking:** Records attendance date and time.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application |
| OpenCV | Face detection and recognition |
| YuNet | Face detection |
| SFace | Face recognition and embeddings |
| MediaPipe | Facial landmark detection |
| NumPy | Mathematical operations |
| Supabase | Cloud database |
| Supabase Storage | Face image storage |
| python-dotenv | Environment variable management |

## 📁 Project Structure

```text
FaceAttendance/
│
├── .env
├── .gitignore
├── app.py
├── requirements.txt
│
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   ├── face_recognition_sface_2021dec.onnx
│   ├── haarcascade_frontalface_default.xml
│   └── face_landmarker.task
│
├── database/
│   ├── __init__.py
│   └── database.py
│
├── engine/
│   ├── __init__.py
│   ├── recognition.py
│   └── liveness.py
│
└── src/
    ├── __init__.py
    └── main.py
```

## 🧠 How It Works

### 1. Student Registration
- Captures a student's face through a webcam.
- Detects the face using YuNet.
- Generates a facial embedding using SFace.
- Extracts facial landmarks using MediaPipe.
- Uploads the face image to Supabase Storage.
- Saves student information and facial data in Supabase.

### 2. Face Recognition
- Captures a live webcam frame.
- Detects the face.
- Aligns and processes the face using SFace.
- Generates a facial embedding.
- Compares it with registered student embeddings using cosine similarity.
- Identifies the student when the similarity score meets the configured threshold.

### 3. Liveness Detection
- MediaPipe detects facial landmarks.
- Eye landmark coordinates are used to calculate the Eye Aspect Ratio (EAR).
- The system detects whether the eyes open and close.
- A detected blink helps verify that the face is live rather than simply a static photograph.

### 4. Attendance Recording
- Recognizes the student after liveness verification.
- Records the student's attendance in Supabase.
- Saves the attendance date, time, and status.
- Prevents duplicate attendance during the session.
- Marks unrecognized students as absent when the session is completed.

## 📐 Mathematical Concepts

This project uses several mathematical concepts:

**1. Euclidean Distance**

Used to calculate the distance between facial landmark coordinates.


**2. Eye Aspect Ratio (EAR)**

Used to detect whether the eyes are open or close

**3. Cosine Similarity**

Used to compare two facial embeddings.

**4. L2 Normalization**

Used to normalize facial embedding vectors.

## 🗄️ Supabase Database

The project uses Supabase for cloud data management.

The application works with these tables:

- `students` – Stores student information and facial embeddings.
- `student_landmarks` – Stores facial landmark information.
- `attendance` – Stores attendance records, dates, times, and status.

Supabase Storage is used to store captured student face images.

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/rekant007/FaceAttendance.git
cd FaceAttendance
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_key
```

Use your appropriate Supabase key and keep your credentials private.

### 5. Download the Required Models

Place all four required model files inside the `models/` directory:

- YuNet face detection model
- SFace face recognition model
- Haar Cascade XML file
- MediaPipe Face Landmarker task model

### 6. Run the Application

From the project root, execute:

```bash
python -m src.main
```

## 🔐 Security Notes

- Keep `.env` out of version control.
- Never expose your Supabase service-role key in frontend code or public repositories.
- Configure appropriate Supabase Row Level Security (RLS) policies.
- Restrict access to stored facial images and biometric data.
- Obtain appropriate consent before collecting or processing facial data.

## 🚀 Future Improvements

- Real-time web-based attendance dashboard.
- Multiple camera support.
- Improved anti-spoofing and liveness verification.
- Attendance reports and analytics.
- Student profile management.
- Export attendance reports to CSV and Excel.
- Improved recognition performance and accuracy.

## 👨‍💻 Developer

**Rekant Gope**

GitHub: [@rekant007](https://github.com/rekant007)

---

**Project:** AI Face Recognition Attendance System

**Built with:** Python · OpenCV · MediaPipe · Supabase

*Turning facial recognition into automated attendance intelligence.*
