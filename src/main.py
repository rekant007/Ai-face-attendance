import cv2
import mediapipe as mp
import uuid
from pathlib import Path

from app import (
    yunet,
    sface,
    haarcascade,
    landmarker,
    blink_detector
)

from database.database import (
    upload_student_image,
    save_student,
    save_landmarks,
    get_all_students,
    mark_attendance,
    mark_session_absentees
)

from engine.recognition import generate_embedding, compare_faces


YUNET_THRESHOLD = 0.363


def get_face_landmarks(frame):

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    return result.face_landmarks[0]


def get_face_crop(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = haarcascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) != 1:
        return None

    x, y, w, h = faces[0]

    padding = 20

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(frame.shape[1], x + w + padding)
    y2 = min(frame.shape[0], y + h + padding)

    return frame[y1:y2, x1:x2]


def register_student(frame):

    try:
        student_id = int(input("Enter Student ID: "))
        name = input("Enter Student Name: ").strip()

        if not name:
            print("Student name cannot be empty.")
            return

        embedding = generate_embedding(frame)

        if embedding is None:
            print("Face not detected. Try again.")
            return

        face_crop = get_face_crop(frame)

        if face_crop is None:
            print("Exactly one face must be visible.")
            return

        landmarks = get_face_landmarks(frame)

        if landmarks is None:
            print("MediaPipe could not detect the face.")
            return

        landmark_data = []

        for index, point in enumerate(landmarks):
            landmark_data.append({
                "point": index,
                "x": point.x,
                "y": point.y,
                "z": point.z
            })

        success, encoded_image = cv2.imencode(
            ".jpg",
            face_crop
        )

        if not success:
            print("Could not encode face image.")
            return

        image_bytes = encoded_image.tobytes()

        storage_path = (
            f"captured_faces/face_{uuid.uuid4().hex}.jpg"
        )

        upload_student_image(storage_path, image_bytes)

        save_student(
            student_id,
            name,
            embedding,
            storage_path
        )

        save_landmarks(
            student_id,
            storage_path,
            landmark_data
        )

        print("\nStudent registered successfully!")
        print("Student ID:", student_id)
        print("Name:", name)

    except ValueError:
        print("Please enter a valid Student ID.")

    except Exception as error:
        print("Registration error:", error)


def recognize_student(frame, students):

    live_embedding = generate_embedding(frame)

    if live_embedding is None:
        return None, 0

    best_student = None
    best_score = -1

    for student in students:

        stored_embedding = student["embedding"]

        score = compare_faces(
            live_embedding,
            stored_embedding
        )

        if score > best_score:
            best_score = score
            best_student = student

    if best_student is not None and best_score >= YUNET_THRESHOLD:
        return best_student, best_score

    return None, best_score


def attendance_mode():

    camera = None

    try:
        students = get_all_students()

        if not students:
            print("No registered students found.")
            return

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            print("Could not open webcam.")
            return

        recognized_in_session = set()
        liveness_passed = False

        blink_detector.reset()

        print("\nAttendance mode started.")
        print("Blink your eyes to pass the liveness check.")
        print("Press Q to stop attendance.")

        while True:

            ret, frame = camera.read()

            if not ret:
                print("Could not read webcam frame.")
                break

            display_frame = frame.copy()

            landmarks = get_face_landmarks(frame)

            if landmarks is None:

                liveness_passed = False
                blink_detector.reset()

                cv2.putText(
                    display_frame,
                    "No Face Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            else:

                if not liveness_passed:

                    blink_detected, ear = blink_detector.update(
                        landmarks
                    )

                    cv2.putText(
                        display_frame,
                        f"EAR: {ear:.3f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2
                    )

                    if blink_detected:
                        liveness_passed = True
                        print("Liveness check passed!")

                    else:
                        cv2.putText(
                            display_frame,
                            "Please Blink Your Eyes",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2
                        )

                if liveness_passed:

                    student, score = recognize_student(
                        frame,
                        students
                    )

                    if student is not None:

                        student_id = student["student_id"]
                        name = student["name"]

                        cv2.putText(
                            display_frame,
                            f"Recognized: {name}",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            display_frame,
                            f"Match: {score:.3f}",
                            (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

                        if student_id not in recognized_in_session:

                            mark_attendance(student_id, "Present")
                            recognized_in_session.add(student_id)

                            print(f"Attendance marked: {name}")

                    else:

                        cv2.putText(
                            display_frame,
                            "Unknown Face",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

            cv2.putText(
                display_frame,
                "Press Q to Stop Attendance",
                (20, display_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            cv2.imshow(
                "Face Recognition Attendance",
                display_frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        absent_count = mark_session_absentees(
            recognized_in_session,
            students
        )

        print("\n========== ATTENDANCE SUMMARY ==========")
        print("Present:", len(recognized_in_session))
        print("Absent:", absent_count)
        print("Total:", len(students))

    except Exception as error:
        print("Attendance error:", error)

    finally:

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()


def main():

    try:

        while True:

            print("\n========== FACE ATTENDANCE SYSTEM ==========")
            print("1. Register Student")
            print("2. Mark Attendance")
            print("3. Exit")

            choice = input("Enter your choice: ").strip()

            if choice == "1":

                camera = cv2.VideoCapture(0)

                if not camera.isOpened():
                    print("Could not open webcam.")
                    continue

                print("Press S to capture student face.")
                print("Press Q to cancel.")

                try:
                    while True:

                        ret, frame = camera.read()

                        if not ret:
                            break

                        cv2.imshow("Student Registration", frame)

                        key = cv2.waitKey(1) & 0xFF

                        if key == ord("s"):
                            register_student(frame)
                            break

                        elif key == ord("q"):
                            break

                finally:
                    camera.release()
                    cv2.destroyAllWindows()

            elif choice == "2":
                attendance_mode()

            elif choice == "3":
                print("Exiting Face Attendance System.")
                break

            else:
                print("Invalid choice. Try again.")

    finally:
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()