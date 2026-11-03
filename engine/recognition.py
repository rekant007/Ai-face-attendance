import cv2
import numpy as np

from app import yunet, sface


def generate_embedding(frame):

    height, width = frame.shape[:2]

    yunet.setInputSize((width, height))

    _, faces = yunet.detect(frame)

    if faces is None or len(faces) == 0:
        return None

    if len(faces) != 1:
        raise ValueError("Please show exactly one face!")

    face = faces[0]

    aligned_face = sface.alignCrop(frame, face)

    embedding = sface.feature(aligned_face)

    embedding = cv2.normalize(embedding, None)

    return embedding.flatten().tolist()


def compare_faces(embedding1, embedding2):

    embedding1 = np.array(
        embedding1, dtype=np.float32
    ).reshape(1, -1)

    embedding2 = np.array(
        embedding2, dtype=np.float32
    ).reshape(1, -1)

    score = sface.match(
        embedding1,
        embedding2,
        cv2.FaceRecognizerSF_FR_COSINE
    )

    return float(score)