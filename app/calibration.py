import os
import glob
import yaml

import cv2 as cv
import numpy as np

settings = {}


def intrinsic():
    cam_ids = settings["cameras"]
    frame_count = settings["frames"]
    cooldown_time = settings["cooldown"]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 100, 0.001)

    rows = settings["checkerboard_rows"] - 1
    cols = settings["checkerboard_columns"] - 1
    world_scaling = settings["checkerboard_box_size"]

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    objp = world_scaling * objp

    width = 0
    height = 0

    for cam_idx, cam in enumerate(cam_ids):
        cooldown = cooldown_time * 10
        start = False
        saved_count = 0

        imgpoints = []
        objpoints = []

        cap = cv.VideoCapture(cam)
        while saved_count < frame_count:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture from one or more cameras.")
                quit()

            if start:
                cooldown -= 1

            if width == 0:
                width = frame.shape[1]
                height = frame.shape[0]

            prev = np.copy(frame)

            if not start:
                cv.putText(
                    prev,
                    "Press SPACEBAR to start collection frames",
                    (10, 30),
                    cv.FONT_HERSHEY_DUPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            else:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                ret, corners = cv.findChessboardCorners(gray, (rows, cols), None)

                if ret:
                    corners = cv.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria
                    )
                    cv.drawChessboardCorners(prev, (rows, cols), corners, ret)

                    if cooldown <= 0:
                        imgpoints.append(corners)
                        objpoints.append(objp)
                        cooldown = cooldown_time * 10
                        saved_count += 1

                cv.putText(
                    prev,
                    f"Cooldown {cooldown}",
                    (10, 30),
                    cv.FONT_HERSHEY_DUPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv.putText(
                    prev,
                    f"Frames {saved_count}",
                    (10, 60),
                    cv.FONT_HERSHEY_DUPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            cv.imshow("Preview", prev)
            match cv.waitKey(1):
                case 27:
                    quit()
                case 32:
                    cooldown = cooldown_time * 10
                    start = True
                case _:
                    continue

        cv.destroyAllWindows()
        rmse, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints, imgpoints, (width, height), None, None
        )

        print(f"cam{cam_idx}:")
        print("  rmse:", rmse)
        print("  camera matrix:\n", mtx)
        print("  distortion coeffs:", dist)

        save_camera_intrinsic(cam_idx, mtx, dist)

    return mtx, dist


def stereo_calibrate():
    cam_ids = settings["cameras"]
    frame_count = settings["frames"]
    cooldown_time = settings["cooldown"]

    cooldown = cooldown_time * 10
    start = False
    saved_count = 0

    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 100, 0.001)

    caps = [cv.VideoCapture(cam_id) for cam_id in cam_ids]

    rows = settings["checkerboard_rows"] - 1
    cols = settings["checkerboard_columns"] - 1
    world_scaling = settings["checkerboard_box_size"]

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    objp = world_scaling * objp

    width = 0
    height = 0

    mtx0, dist0 = read_camera_intrinsic(0)
    mtx1, dist1 = read_camera_intrinsic(1)

    imgpoints = [[], []]
    objpoints = []
    while saved_count < frame_count:
        frames = []
        for cam_idx, cam_id in enumerate(cam_ids):
            ret, frame = caps[cam_idx].read()
            if ret:
                frames.append(frame)
            else:
                print("Failed to capture from one or more cameras.")
                quit()

        if width == 0:
            width = frame.shape[1]
            height = frame.shape[0]

        if start:
            cooldown -= 1

        prev = []
        prevs = [np.copy(frame) for frame in frames]
        if not start:
            prev = np.hstack((prevs[0], prevs[1]))
            cv.putText(
                prev,
                "Press SPACEBAR to start collection frames",
                (10, 30),
                cv.FONT_HERSHEY_DUPLEX,
                1,
                (0, 0, 255),
                2,
            )

        else:
            grays = [cv.cvtColor(frame, cv.COLOR_BGR2GRAY) for frame in frames]

            rets = []
            corners = []
            for i, gray in enumerate(grays):
                ret, new_corners = cv.findChessboardCorners(gray, (rows, cols), None)
                rets.append(ret)
                corners.append(new_corners)

            if all(rets):
                for i, gray in enumerate(grays):
                    corners[i] = cv.cornerSubPix(
                        gray, corners[i], (11, 11), (-1, -1), criteria
                    )

                p0 = [corner[0, 0].astype(np.int32) for corner in corners]

                for i, prev in enumerate(prevs):
                    cv.putText(
                        prev,
                        "O",
                        (p0[i][0], p0[i][1]),
                        cv.FONT_HERSHEY_DUPLEX,
                        1,
                        (0, 0, 255),
                        1,
                    )
                    cv.drawChessboardCorners(prev, (rows, cols), corners[i], rets[i])

                if cooldown <= 0:
                    imgpoints[0].append(corners[0])
                    imgpoints[1].append(corners[1])
                    objpoints.append(objp)
                    cooldown = cooldown_time * 10
                    saved_count += 1

            prev = np.hstack((prevs[0], prevs[1]))
            cv.putText(
                prev,
                f"Cooldown {cooldown}",
                (10, 30),
                cv.FONT_HERSHEY_DUPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv.putText(
                prev,
                f"Frames {saved_count}",
                (10, 60),
                cv.FONT_HERSHEY_DUPLEX,
                1,
                (0, 255, 0),
                2,
            )

        cv.imshow("Preview", prev)
        match cv.waitKey(1):
            case 27:
                quit()
            case 32:
                cooldown = cooldown_time * 10
                start = True
            case _:
                continue

    cv.destroyAllWindows()
    ret, CM1, dist0, CM2, dist1, R1, T1, E, F = cv.stereoCalibrate(
        objpoints,
        imgpoints[0],
        imgpoints[1],
        mtx0,
        dist0,
        mtx1,
        dist1,
        (width, height),
        criteria=criteria,
        flags=cv.CALIB_FIX_INTRINSIC,
    )

    print("rmse: ", ret)
    cv.destroyAllWindows()

    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0.0, 0.0, 0.0]).reshape((3, 1))

    save_camera_extrinsic(0, R0, T0)
    save_camera_extrinsic(1, R1, T1)


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


def save_camera_intrinsic(cam_id, mtx, dist):
    if not os.path.exists("camera_parameters"):
        os.mkdir("camera_parameters")

    out_filename = os.path.join("camera_parameters", f"cam{cam_id}_intrinsic.dat")
    outf = open(out_filename, "w")

    outf.write("intrinsic:\n")
    for l in mtx:
        for r in l:
            outf.write(str(r) + " ")
        outf.write("\n")

    outf.write("distortion:\n")
    for en in dist[0]:
        outf.write(str(en) + " ")
    outf.write("\n")


def read_camera_intrinsic(cam_id):
    inf = open(f"camera_parameters/cam{cam_id}_intrinsic.dat", "r")

    cmtx = []
    dist = []

    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(row) for row in line]
        cmtx.append(line)

    line = inf.readline()
    line = inf.readline().split()
    line = [float(row) for row in line]
    dist.append(line)

    return np.array(cmtx), np.array(dist)


def save_camera_extrinsic(cam_id, R, T):
    if not os.path.exists("camera_parameters"):
        os.mkdir("camera_parameters")

    cam_rot_trans_filename = os.path.join(
        "camera_parameters", "cam{cam_id}_rot_trans.dat"
    )
    outf = open(cam_rot_trans_filename, "w")

    outf.write("R:\n")
    for l in R:
        for en in l:
            outf.write(str(en) + " ")
        outf.write("\n")

    outf.write("T:\n")
    for l in T:
        for en in l:
            outf.write(str(en) + " ")
        outf.write("\n")
    outf.close()


def read_camera_extrinsic(cam_id):
    inf = open(f"camera_parameters/cam{cam_id}_rot_trans.dat", "r")

    inf.readline()
    rot = []
    trans = []
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        rot.append(line)

    inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        trans.append(line)

    inf.close()
    return np.array(rot), np.array(trans)
