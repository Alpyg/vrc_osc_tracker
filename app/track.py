import time
from math import atan2, cos, sin
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.python import solutions
from utils import draw_landmarks_on_image
from osc import handle_osc
import calibration


model_path = "./models/pose_landmarker_lite.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def recalculate_world_space(landmarks):
    head = landmarks[solutions.pose.PoseLandmark.NOSE]
    left_hip = landmarks[solutions.pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[solutions.pose.PoseLandmark.RIGHT_HIP]

    flip_z = 1 if right_hip.x > left_hip.x else -1
    x_axis = [right_hip.x - left_hip.x, 0, (right_hip.z - left_hip.z) * flip_z]
    x_axis /= np.linalg.norm(x_axis)
    angle = atan2(x_axis[2], x_axis[0])

    for lm in landmarks:
        lm.x -= head.x
        lm.y -= head.y
        lm.z -= head.z

        new_x = cos(angle) * lm.x + sin(angle) * lm.z
        new_z = -sin(angle) * lm.x + cos(angle) * lm.z

        lm.x = new_x
        lm.z = new_z


def result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global world_landmarks

    for landmarks in result.pose_world_landmarks:
        recalculate_world_space(landmarks)

    annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)

    cv.imshow("VRC OSC Tracker", annotated_image)

    handle_osc(result)


options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=model_path, delegate=BaseOptions.Delegate.GPU
    ),
    running_mode=VisionRunningMode.LIVE_STREAM,
    output_segmentation_masks=False,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    num_poses=1,
    result_callback=result,
)


def start():
    cap = cv.VideoCapture(calibration.settings["cameras"][1])

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

            if cv.waitKey(10) & 0xFF == 27:
                break

    cap.release()
