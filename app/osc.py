import mediapipe as mp
import numpy as np
from mediapipe.python import solutions
from mediapipe.tasks.python.components.containers.landmark import Landmark
from pythonosc import udp_client

from tracker import Tracker


osc_ip = "10.147.20.18"
osc_port = 9000
osc = udp_client.SimpleUDPClient(osc_ip, osc_port)

trackers = {
    "head": Tracker(namespace="head", offset=(0, 1.73, -0.2), scale=(1, 0, 1)),
    "hip": Tracker(namespace="1", offset=(0, 0.6, 0), scale=(1, 1, 1)),
    "left_foot": Tracker(namespace="2", offset=(-0.05, 0.3, 0.15), scale=(1, 1.5, 3)),
    "right_foot": Tracker(namespace="3", offset=(0.05, 0.3, 0.15), scale=(1, 1.5, 3)),
}


def center_landmark(left: Landmark, right: Landmark):
    return Landmark(
        (left.x + right.x) / 2.0,
        (left.y + right.y) / 2.0,
        (left.z + right.z) / 2.0,
        (left.visibility + right.visibility) / 2.0,
        (left.presence + right.presence) / 2.0,
    )


def handle_osc(result):
    global trackers

    for landmarks in result.pose_world_landmarks:
        head = landmarks[solutions.pose.PoseLandmark.NOSE]
        hip = center_landmark(
            landmarks[solutions.pose.PoseLandmark.LEFT_HIP],
            landmarks[solutions.pose.PoseLandmark.RIGHT_HIP],
        )
        left_foot = landmarks[solutions.pose.PoseLandmark.LEFT_ANKLE]
        right_foot = landmarks[solutions.pose.PoseLandmark.RIGHT_ANKLE]

        trackers["head"].update_position(head)
        trackers["hip"].update_position(hip)
        trackers["left_foot"].update_position(left_foot)
        trackers["right_foot"].update_position(right_foot)

        for tracker in trackers.values():
            tracker.send_osc(osc)
