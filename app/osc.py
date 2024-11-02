import mediapipe as mp
import numpy as np
from mediapipe.python import solutions
from pythonosc import udp_client
from math import atan2, asin, degrees, sqrt

from tracker import Tracker


osc_ip = "127.0.0.1"
osc_port = 9000
osc = udp_client.SimpleUDPClient(osc_ip, osc_port)

trackers = {
    "head": Tracker(namespace="head", offset=(0, 0, -0.2), scale=(-1, 0, -1)),
    "left_foot": Tracker(namespace="1", offset=(0, -1.2, 0), scale=(-1, 1.5, -1)),
    "right_foot": Tracker(namespace="2", offset=(0, -1.2, 0), scale=(-1, 1.5, -1)),
}


def handle_osc(result, base_alpha=0.1):
    global trackers

    for landmarks in result.pose_world_landmarks:
        head = landmarks[solutions.pose.PoseLandmark.NOSE]
        left_foot = landmarks[solutions.pose.PoseLandmark.LEFT_ANKLE]
        right_foot = landmarks[solutions.pose.PoseLandmark.RIGHT_ANKLE]

        trackers["head"].update_position(head)
        trackers["left_foot"].update_position(left_foot)
        trackers["right_foot"].update_position(right_foot)

        for tracker in trackers.values():
            tracker.send_osc(osc)
