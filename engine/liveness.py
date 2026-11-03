
import math
import time


LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


class BlinkDetector:

    def __init__(
        self,
        open_threshold=0.22,
        closed_threshold=0.18,
        cooldown=0.5
    ):
        self.open_threshold = open_threshold
        self.closed_threshold = closed_threshold
        self.cooldown = cooldown

        self.eye_was_open = False
        self.eye_was_closed = False
        self.last_blink_time = 0

    def distance(self, point1, point2):

        return math.sqrt(
            (point1.x - point2.x) ** 2 +
            (point1.y - point2.y) ** 2
        )

    def calculate_ear(self, landmarks, eye_indices):

        p1 = landmarks[eye_indices[0]]
        p2 = landmarks[eye_indices[1]]
        p3 = landmarks[eye_indices[2]]
        p4 = landmarks[eye_indices[3]]
        p5 = landmarks[eye_indices[4]]
        p6 = landmarks[eye_indices[5]]

        vertical_1 = self.distance(p2, p6)
        vertical_2 = self.distance(p3, p5)

        horizontal = self.distance(p1, p4)

        if horizontal == 0:
            return 0.0

        ear = (vertical_1 + vertical_2) / (2 * horizontal)

        return ear

    def get_average_ear(self, landmarks):

        left_ear = self.calculate_ear(
            landmarks,
            LEFT_EYE
        )

        right_ear = self.calculate_ear(
            landmarks,
            RIGHT_EYE
        )

        return (left_ear + right_ear) / 2

    def update(self, landmarks):

        ear = self.get_average_ear(landmarks)

        blink_detected = False

        current_time = time.monotonic()

        if ear >= self.open_threshold:

            if self.eye_was_closed:

                if (
                    current_time - self.last_blink_time
                    >= self.cooldown
                ):
                    blink_detected = True
                    self.last_blink_time = current_time

                self.eye_was_closed = False

            self.eye_was_open = True

        elif ear <= self.closed_threshold:

            if self.eye_was_open:
                self.eye_was_closed = True

        return blink_detected, ear

    def reset(self):

        self.eye_was_open = False
        self.eye_was_closed = False
        self.last_blink_time = 0
