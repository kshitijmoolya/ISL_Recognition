import cv2
import csv
import os
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def extract_landmarks(hand_landmarks):

    landmarks = []

    for landmark in hand_landmarks:
        landmarks.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return np.array(landmarks, dtype=np.float32)


def extract_two_hands(results):

    left_hand = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)

    if results.hand_landmarks:

        for i, hand_landmarks in enumerate(
            results.hand_landmarks
        ):

            features = extract_landmarks(hand_landmarks)

            handedness = results.handedness[i][0].category_name

            if handedness == "Left":
                left_hand = features

            elif handedness == "Right":
                right_hand = features

    return np.concatenate([left_hand, right_hand])
def normalize_hand(features):
    """
    Normalize 63 hand features relative to wrist (landmark 0).
    """

    # Convert flat array into 21 rows × 3 coordinates
    landmarks = features.reshape(21, 3)

    # Wrist coordinates
    wrist = landmarks[0].copy()

    # Make coordinates relative to wrist
    landmarks = landmarks - wrist

    # Flatten back to 63 values
    return landmarks.flatten()
# -----------------------------
# Model
# -----------------------------
MODEL_PATH = "hand_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)

dataset_file = "isl_dataset.csv"

if not os.path.exists(dataset_file):

    with open(dataset_file, "w", newline="") as file:

        writer = csv.writer(file)

        header = ["label"]

        for i in range(126):
            header.append(f"feature_{i}")

        writer.writerow(header)
current_label = "A"
# -----------------------------
# Camera
# -----------------------------
camera = cv2.VideoCapture(0)

frame_timestamp_ms = 0

while True:

    success, frame = camera.read()

    if not success:
        break

    # BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hands
    results = detector.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )
    features = extract_two_hands(results)
    left_hand = normalize_hand(features[:63])
    right_hand = normalize_hand(features[63:])
    normalized_features = np.concatenate([
    left_hand,
    right_hand
])
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        row = [current_label] + normalized_features.tolist()
        with open(dataset_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)
        print(f"Saved sample for sign: {current_label}")

    print(features.shape)

    frame_timestamp_ms += 33


    # -----------------------------
    # Extract 21 landmarks
    # -----------------------------
    if results.hand_landmarks:

        for hand_number, hand_landmarks in enumerate(
            results.hand_landmarks
        ):

            print("\n==============================")
            print(f"Hand {hand_number + 1}")
            print("==============================")

            for landmark_number, landmark in enumerate(
                hand_landmarks
            ):

                print(
                    f"Landmark {landmark_number:2d} : "
                    f"x = {landmark.x:.4f}, "
                    f"y = {landmark.y:.4f}, "
                    f"z = {landmark.z:.4f}"
                )


    # -----------------------------
    # Draw landmarks
    # -----------------------------
    if results.hand_landmarks:

        h, w, _ = frame.shape

        for hand_landmarks in results.hand_landmarks:

            # Draw points
            for landmark in hand_landmarks:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Draw connections
            connections = (
                vision.HandLandmarksConnections.HAND_CONNECTIONS
            )

            for connection in connections:

                start = hand_landmarks[connection.start]
                end = hand_landmarks[connection.end]

                x1 = int(start.x * w)
                y1 = int(start.y * h)

                x2 = int(end.x * w)
                y2 = int(end.y * h)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )


    # Show camera
    cv2.imshow("ISL Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        row = [current_label] + normalized_features.tolist()
        with open(dataset_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)
        print(f"Saved sample for sign: {current_label}")
    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
detector.close()
