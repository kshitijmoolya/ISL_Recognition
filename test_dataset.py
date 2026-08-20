import cv2
import mediapipe as mp
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------------
# SETTINGS
# -----------------------------------

MODEL_PATH = "hand_landmarker.task"

DATASET_FOLDER = r"C:\Users\Admin\Desktop\ISL_Recognition\Testing\A"


# -----------------------------------
# CREATE MEDIAPIPE DETECTOR
# -----------------------------------

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(
    options
)


# -----------------------------------
# GET FIRST 10 IMAGES
# -----------------------------------

images = [
    file for file in os.listdir(DATASET_FOLDER)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]

images.sort()

images = images[:10]


successful = 0
failed = 0


# -----------------------------------
# PROCESS IMAGES
# -----------------------------------

for filename in images:

    image_path = os.path.join(
        DATASET_FOLDER,
        filename
    )

    frame = cv2.imread(image_path)

    if frame is None:

        print(f"{filename} → Could not read image")

        failed += 1
        continue


    # BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # Detect hands
    results = detector.detect(mp_image)


    # -----------------------------------
    # CHECK RESULT
    # -----------------------------------

    if results.hand_landmarks:

        number_of_hands = len(
            results.hand_landmarks
        )

        print(
            f"{filename} → "
            f"{number_of_hands} hand(s) detected ✓"
        )

        successful += 1


        # Draw landmarks
        h, w, _ = frame.shape

        for hand_landmarks in results.hand_landmarks:

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


    else:

        print(
            f"{filename} → "
            f"No hand detected ✗"
        )

        failed += 1


    # Display image
    cv2.imshow(
        "Dataset Test",
        frame
    )


    # Wait 500 ms
    key = cv2.waitKey(500)

    if key == ord("q"):
        break


# -----------------------------------
# RESULTS
# -----------------------------------

print("\n==============================")
print("DATASET TEST RESULTS")
print("==============================")

print(f"Images tested : {len(images)}")
print(f"Successful    : {successful}")
print(f"Failed        : {failed}")

if len(images) > 0:

    success_rate = (
        successful / len(images)
    ) * 100

    print(
        f"Success rate  : "
        f"{success_rate:.2f}%"
    )


cv2.destroyAllWindows()

detector.close()
