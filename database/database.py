
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from supabase import create_client


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials missing!")

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

bucket = supabase.storage.from_("student_face")

print("Supabase connected!")


def upload_student_image(storage_path, image_bytes):

    return bucket.upload(
        path=storage_path,
        file=image_bytes,
        file_options={
            "content-type": "image/jpeg",
            "upsert": "false"
        }
    )


def save_student(student_id, name, embedding, storage_path):

    data = {
        "student_id": student_id,
        "name": name,
        "embedding": embedding,
        "storage_path": storage_path
    }

    response = (
        supabase
        .table("students")
        .upsert(data)
        .execute()
    )

    return response.data


def save_landmarks(student_id, storage_path, landmarks):

    data = {
        "student_id": student_id,
        "storage_path": storage_path,
        "landmarks": landmarks,
        "landmark_count": len(landmarks)
    }

    response = (
        supabase
        .table("student_landmarks")
        .insert(data)
        .execute()
    )

    return response.data


def get_all_students():

    response = (
        supabase
        .table("students")
        .select("student_id, name, embedding")
        .execute()
    )

    return response.data or []


def mark_attendance(student_id, status="Present"):

    today = datetime.now(
        ZoneInfo("Asia/Kolkata")
    ).date().isoformat()

    data = {
        "student_id": student_id,
        "attendance_date": today,
        "status": status
    }

    response = (
        supabase
        .table("attendance")
        .upsert(
            data,
            on_conflict="student_id,attendance_date",
            ignore_duplicates=True
        )
        .execute()
    )

    return response.data


def mark_session_absentees(recognized_ids, students):

    absent_count = 0

    for student in students:

        student_id = student["student_id"]

        if student_id not in recognized_ids:

            mark_attendance(
                student_id,
                "Absent"
            )

            absent_count += 1

    return absent_count


print("Database functions ready!")
