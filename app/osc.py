import mediapipe as mp
import numpy as np
from mediapipe.python import solutions
from pythonosc import udp_client
from math import atan2, asin, degrees, sqrt

from .config import config


osc_ip = "127.0.0.1"
osc_port = 9000
osc = udp_client.SimpleUDPClient(osc_ip, osc_port)

previous_landmarks = {
    "head": None,
    "left_foot": None,
    "right_foot": None,
    "left_heel": None,
    "right_heel": None,
    "left_toe": None,
    "right_toe": None,
}


def lerp(value1, value2, alpha):
    return value1 * (1 - alpha) + value2 * alpha


def adaptive_alpha(prev, current, base_alpha=0.1, distance_threshold=0.1):
    distance = sqrt(
        (prev.x - current.x) ** 2
        + (prev.y - current.y) ** 2
        + (prev.z - current.z) ** 2
    )
    alpha = min(1, base_alpha + (distance / distance_threshold) * base_alpha)
    return alpha


def get_lerped_position(previous_landmark, landmark, alpha):
    return Point(
        (
            lerp(
                previous_landmark.x,
                landmark.x,
                adaptive_alpha(previous_landmark, landmark, alpha),
            )
            if previous_landmark
            else landmark.x
        ),
        (
            lerp(
                previous_landmark.y,
                landmark.y,
                adaptive_alpha(previous_landmark, landmark, alpha),
            )
            if previous_landmark
            else landmark.x
        ),
        (
            lerp(
                previous_landmark.z,
                landmark.z,
                adaptive_alpha(previous_landmark, landmark, alpha),
            )
            if previous_landmark
            else landmark.x
        ),
    )


def handle_osc(result, base_alpha=0.1):
    global previous_landmarks

    for landmarks in result.pose_world_landmarks:
        # Get positions for left and right feet
        head = landmarks[solutions.pose.PoseLandmark.NOSE]
        left_foot = landmarks[solutions.pose.PoseLandmark.LEFT_ANKLE]
        right_foot = landmarks[solutions.pose.PoseLandmark.RIGHT_ANKLE]
        left_heel = landmarks[solutions.pose.PoseLandmark.LEFT_HEEL]
        right_heel = landmarks[solutions.pose.PoseLandmark.RIGHT_HEEL]
        left_toe = landmarks[solutions.pose.PoseLandmark.LEFT_FOOT_INDEX]
        right_toe = landmarks[solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX]

        # Interpolate positions
        lerped_head = get_lerped_position(previous_landmarks["head"], head, base_alpha)
        lerped_left_foot = get_lerped_position(
            previous_landmarks["left_foot"], left_foot, base_alpha
        )
        lerped_right_foot = get_lerped_position(
            previous_landmarks["right_foot"], right_foot, base_alpha
        )
        lerped_left_heel = get_lerped_position(
            previous_landmarks["left_heel"], left_heel, base_alpha
        )
        lerped_right_heel = get_lerped_position(
            previous_landmarks["right_heel"], right_heel, base_alpha
        )
        lerped_left_toe = get_lerped_position(
            previous_landmarks["left_toe"], left_toe, base_alpha
        )
        lerped_right_toe = get_lerped_position(
            previous_landmarks["right_toe"], right_toe, base_alpha
        )

        # Send interpolated positions over OSC
        osc.send_message(
            "/tracking/trackers/head/position",
            [-lerped_head.x, lerped_head.y, -lerped_head.z],
        )
        osc.send_message(
            "/tracking/trackers/1/position",
            [-lerped_left_foot.x, -1.2 - lerped_left_foot.y * 1.5, -lerped_left_foot.z],
        )
        osc.send_message(
            "/tracking/trackers/2/position",
            [
                -lerped_right_foot.x,
                -1.2 - lerped_right_foot.y * 1.5,
                -lerped_right_foot.z,
            ],
        )

        # # Calculate interpolated rotations
        # lerped_left_pitch, lerped_left_yaw, lerped_left_roll = calculate_euler_angles(
        #     lerped_left_foot,
        #     lerped_left_heel,
        #     lerped_left_toe,
        # )
        # lerped_right_pitch, lerped_right_yaw, lerped_right_roll = (
        #     calculate_euler_angles(
        #         lerped_right_foot,
        #         lerped_right_heel,
        #         lerped_right_toe,
        #     )
        # )
        #
        # # Send interpolated rotations over OSC
        # osc.send_message(
        #     "/tracking/trackers/1/rotation",
        #     [lerped_left_pitch, lerped_left_yaw, lerped_left_roll],
        # )
        # osc.send_message(
        #     "/tracking/trackers/2/rotation",
        #     [lerped_right_pitch, lerped_right_yaw, lerped_right_roll],
        # )

        # Update previous landmarks for the next frame
        previous_landmarks = {
            "head": lerped_head,
            "left_foot": lerped_left_foot,
            "right_foot": lerped_right_foot,
            "left_heel": lerped_left_heel,
            "right_heel": lerped_right_heel,
            "left_toe": lerped_right_toe,
            "right_toe": lerped_right_toe,
        }


def calculate_euler_angles(ankle, heel, toe):
    # Calculate direction vectors for foot orientation
    foot_forward = np.array([toe.x - heel.x, toe.y - heel.y, toe.z - heel.z])
    foot_side = np.array([ankle.x - heel.x, ankle.y - heel.y, ankle.z - heel.z])

    # Normalize direction vectors
    forward_norm = foot_forward / np.linalg.norm(foot_forward)
    side_norm = foot_side / np.linalg.norm(foot_side)

    # Calculate yaw, pitch, roll
    yaw = atan2(forward_norm[0], forward_norm[2])  # Rotation around Y-axis
    pitch = asin(-forward_norm[1])  # Rotation around X-axis
    roll = atan2(side_norm[1], side_norm[0])  # Rotation around Z-axis

    # Convert to degrees
    return degrees(pitch), degrees(yaw), -degrees(roll)


class Point:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
