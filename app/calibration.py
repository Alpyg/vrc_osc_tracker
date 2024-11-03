import os
import yaml

import cv2 as cv
import numpy as np

settings = {}


def frames():
    if not os.path.exists("frames"):
        os.mkdir("frames")

    cam_ids = settings["cameras"]
    frame_count = settings["frames"]
    cooldown_time = settings["cooldown"]

    cooldown = cooldown_time
    start = False
    saved_count = 0

    caps = [cv.VideoCapture(cam_id) for cam_id in cam_ids]

    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        if len(frames) != len(caps):
            print("Failed to capture from one or more cameras.")
        else:
            for i, frame in enumerate(frames):
                cv.imshow(f"Camera {cam_ids[i]}", cv.cvtColor(frame, cv.COLOR_RGB2BGR))

        # TODO: resize?

        # preview = np.copy(frames[0])
        #
        # if not start:
        #     cv.putText(
        #         preview,
        #         "Press SPACEBAR to start collection frames",
        #         (50, 50),
        #         cv.FONT_HERSHEY_COMPLEX,
        #         1,
        #         (0, 0, 255),
        #         1,
        #     )
        #
        # cv.imshow("Preview", preview)

        k = cv.waitKey(1)
        match k:
            case 27:
                quit()
            case 32:
                start = True
            case _:
                continue


def parse_settings(filename):
    global settings

    if not os.path.exists(filename):
        print("File does not exist:", filename)
        quit()

    print("Using for calibration:", filename)

    with open(filename) as f:
        settings = yaml.safe_load(f)

    if "cameras" not in settings.keys():
        print("`cameras` key was not found in the settings file.")
        quit()
