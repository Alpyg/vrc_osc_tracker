import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .utils import draw_landmarks_on_image
from .osc import handle_osc


model_path = "./models/pose_landmarker_lite.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

world_landmarks = None
annotated_image = np.zeros((480, 640, 3), dtype=np.uint8)


def result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global annotated_image, world_landmarks

    world_landmarks = result.pose_world_landmarks

    annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)

    handle_osc(result)


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    output_segmentation_masks=False,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    num_poses=1,
    result_callback=result,
)


def start_tracking():
    global annotated_image

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open capture device")
        exit(1)

    start = time.time_ns()
    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame_data = cap.read()

            if not ret:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_data)
            landmarker.detect_async(mp_image, time.time_ns() - start)

            time.sleep(1 / 30)

    cap.release()
