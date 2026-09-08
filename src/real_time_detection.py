"""Run hybrid EAR and TensorFlow Lite drowsiness detection on Raspberry Pi."""

from __future__ import annotations

import os
import time
from collections import deque

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from scipy.spatial import distance
from tflite_runtime.interpreter import Interpreter


CALIBRATION_SECONDS = 4.0
SMOOTHING_WINDOW = 7
EYE_CLOSE_SECONDS = 4.0
EAR_FALL_FRACTION = 0.78
EAR_MINIMUM = 0.16
MINIMUM_CALIBRATION_FRAMES = 6
ML_CONFIDENCE_THRESHOLD = 0.7
MODEL_SIZE = 244
SPEAKER_PIN = 18
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]


def eye_aspect_ratio(indices: list[int], landmarks: list[tuple[int, int]]) -> float:
    p1, p2, p3, p4, p5, p6 = [landmarks[index] for index in indices]
    horizontal = distance.euclidean(p1, p4)
    if horizontal < 1e-6:
        return 0.0
    return (distance.euclidean(p2, p6) + distance.euclidean(p3, p5)) / (2.0 * horizontal)


def play_alert(pwm: GPIO.PWM) -> None:
    for _ in range(3):
        pwm.start(80)
        time.sleep(0.2)
        pwm.stop()
        time.sleep(0.1)
        pwm.start(50)
        time.sleep(0.5)
        pwm.stop()
        time.sleep(0.2)


def main() -> None:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SPEAKER_PIN, GPIO.OUT)
    pwm = GPIO.PWM(SPEAKER_PIN, 440)

    interpreter = Interpreter(model_path="driver_drowsiness_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    camera = Picamera2()
    camera.configure(camera.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)}))
    camera.start()
    time.sleep(0.5)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    ear_history: deque[float] = deque(maxlen=SMOOTHING_WINDOW)
    calibration: list[float] = []
    calibration_started = time.time()
    calibrated = False
    ear_threshold = 0.25
    eye_closed_time = 0.0
    last_timestamp = time.time()

    try:
        while True:
            frame = camera.capture_array()
            now = time.time()
            elapsed = max(1e-3, now - last_timestamp)
            last_timestamp = now
            result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            status = "No face"
            color = (128, 128, 128)

            if result.multi_face_landmarks:
                height, width, _ = frame.shape
                points = [
                    (int(mark.x * width), int(mark.y * height))
                    for mark in result.multi_face_landmarks[0].landmark
                ]
                average_ear = (
                    eye_aspect_ratio(LEFT_EYE, points) + eye_aspect_ratio(RIGHT_EYE, points)
                ) / 2.0
                ear_history.append(average_ear)
                smoothed_ear = float(np.mean(ear_history))

                if not calibrated:
                    if now - calibration_started <= CALIBRATION_SECONDS:
                        if average_ear > 0.22:
                            calibration.append(average_ear)
                        status, color = "Calibrating...", (0, 255, 255)
                    else:
                        if len(calibration) >= MINIMUM_CALIBRATION_FRAMES:
                            ear_threshold = max(
                                EAR_MINIMUM,
                                float(np.median(calibration)) * EAR_FALL_FRACTION,
                            )
                        calibrated = True

                eye_closed_time = (
                    eye_closed_time + elapsed
                    if smoothed_ear < ear_threshold
                    else max(0.0, eye_closed_time - elapsed * 0.5)
                )

                resized = cv2.resize(frame, (MODEL_SIZE, MODEL_SIZE))
                channels = input_details[0]["shape"][-1]
                processed = (
                    np.expand_dims(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), axis=-1)
                    if channels == 1
                    else cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                )
                model_input = np.expand_dims(processed.astype("float32") / 255.0, axis=0)
                interpreter.set_tensor(input_details[0]["index"], model_input)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]["index"])[0]
                predicted_class = int(np.argmax(prediction))
                confidence = float(np.max(prediction))

                drowsy = eye_closed_time >= EYE_CLOSE_SECONDS or (
                    predicted_class == 1 and confidence > ML_CONFIDENCE_THRESHOLD
                )
                if calibrated and drowsy:
                    status, color = "DROWSINESS DETECTED", (0, 0, 255)
                    play_alert(pwm)
                    eye_closed_time = 0.0
                elif calibrated:
                    status, color = "ALERT", (0, 200, 0)

                cv2.putText(frame, f"EAR: {smoothed_ear:.2f} (TH {ear_threshold:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"ML: {predicted_class}, confidence: {confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)

            cv2.putText(frame, status, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow("Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.stop()
        face_mesh.close()
        cv2.destroyAllWindows()
        pwm.stop()
        GPIO.cleanup()


if __name__ == "__main__":
    main()

