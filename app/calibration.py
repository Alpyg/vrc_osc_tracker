import os
import glob
import yaml

import cv2 as cv
import numpy as np

settings = {}


def frames():
    if not os.path.exists("frames"):
        os.mkdir("frames")
    else:
        for file in os.listdir("frames"):
            file_path = os.path.join("frames", file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

    cam_ids = settings["cameras"]
    frame_count = settings["frames"]
    cooldown_time = settings["cooldown"]

    cooldown = cooldown_time * 10
    start = False
    saved_count = 0

    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                print("Failed to capture from one or more cameras.")
                quit()

        prev1 = np.copy(frames[0])
        prev2 = np.copy(frames[1])

        if not start:
            cv.putText(
                prev1,
                "Press SPACEBAR to start collection frames",
                (10, 30),
                cv.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 0, 255),
                1,
            )
            cv.putText(
                prev2,
                "Press SPACEBAR to start collection frames",
                (10, 30),
                cv.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 0, 255),
                1,
            )

        if start:
            cooldown -= 1
            cv.putText(
                prev1,
                f"Cooldown: {cooldown}",
                (25, 25),
                cv.FONT_HERSHEY_DUPLEX,
                1,
                (0, 255, 0),
                1,
            )
            cv.putText(
                prev1,
                f"Frame: {saved_count}",
                (25, 50),
                cv.FONT_HERSHEY_DUPLEX,
                1,
                (0, 255, 0),
                1,
            )
            cv.putText(
                prev2,
                f"Cooldown: {cooldown}",
                (25, 25),
                cv.FONT_HERSHEY_DUPLEX,
                1,
                (0, 255, 0),
                1,
            )
            cv.putText(
                prev2,
                f"Frame: {saved_count}",
                (25, 50),
                cv.FONT_HERSHEY_DUPLEX,
                1,
                (0, 255, 0),
                1,
            )

            for i, frame in enumerate(frames):
                if cooldown <= 0:
                    savename = os.path.join("frames", f"cam{i}_{saved_count}.png")
                    cv.imwrite(savename, frame)
                    saved_count += 1
                    cooldown = cooldown_time * 10

        cv.imshow("Preview1", prev1)
        cv.imshow("Preview2", prev2)

        k = cv.waitKey(1)
        match k:
            case 27:
                quit()
            case 32:
                start = True
            case _:
                continue


def intrinsic():
    cam_ids = settings["cameras"]
    frame_count = settings["frames"]
    cooldown_time = settings["cooldown"]

    cooldown = cooldown_time * 10
    start = False
    saved_count = 0

    caps = [cv.VideoCapture(cam_id) for cam_id in cam_ids]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 100, 0.001)

    rows = settings["checkerboard_rows"] - 1
    cols = settings["checkerboard_columns"] - 1
    world_scaling = settings["checkerboard_box_size"]

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    objp = world_scaling * objp

    width = 0
    height = 0

    cam_imagepoints = []
    cam_objpoints = []
    while save_count < settings["frames"]:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                print("Failed to capture from one or more cameras.")
                quit()

        prev = []
        prevs = []
        for frame in frames:
            prevs.append(np.copy(frame))

        if not start:
            prev = np.hstack((prev[0], prev[1]))
            width = frames[0].shape[1]
            height = frames[0].shape[0]
            cv.putText(
                prev,
                "Press SPACEBAR to start collection frames",
                (10, 30),
                cv.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 0, 255),
                1,
            )

        else if start:
            cooldown -= 1

            grays = []
            for frame in frames:
                grays.append(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))

            rets = []
            corners = []
            for gray in grays:
                ret, corners = cv.findChessboardCorners(gray, (rows, cols), None)
                rets.append(ret)
                corners.append(corners)

            if all(rets):
                conv_size = (11, 11)

                for i, gray in enumerate(grays):
                    corners[i] = cv.cornerSubPix(
                        gray, corners[i], conv_size, (-1, -1), criteria
                    )
                    cv.drawChessboardCorners(prevs[i], (rows, cols), corners[i], rets[i])

                if cooldown <= 0:
                    cam_imagepoints[cam_idx].append(corners)
                    cam_objpoints[cam_idx].append(objp)

            prev = np.hstack((prev[0], prev[1]))

        cv.imshow("Preview", prev)
        if cv.waitKey(0) & 0xFF == 27:
            break

    cv.destroyAllWindows()
    for i, cam_id in enumerate(cam_ids):
        rmse, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            cam_objpoints[i], cam_imgpoints[i], (width, height), None, None
        )

        print(f"cam{cam_id}:")
        print("  rmse:", rmse)
        print("  camera matrix:\n", mtx)
        print("  distortion coeffs:", dist)

        save_camera_intrinsic(cam_id, mtx, dist)

    return mtx, dist


def stereo_calibrate():
    cam_ids = settings["cameras"]
    frames = [sorted(glob.glob(f"frames/cam{cam_id}*")) for cam_id in cam_ids]

    mtx0, dist0 = read_camera_intrinsic(cam_ids[0])
    mtx1, dist1 = read_camera_intrinsic(cam_ids[0])  # TODO: Replace with correct index

    images = [[cv.imread(imname, 1) for imname in cam_frames] for cam_frames in frames]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 100, 0.001)

    rows = settings["checkerboard_rows"] - 1
    cols = settings["checkerboard_columns"] - 1
    world_scaling = settings["checkerboard_box_size"]

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    objp = world_scaling * objp

    width = images[0][0].shape[1]
    height = images[0][0].shape[0]

    imgpoints_left = []
    imgpoints_right = []
    objpoints = []

    for frame0, frame1 in zip(images[0], images[0]):  # TODO: Replace with correct index
        gray1 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, cols), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, cols), None)

        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            p0_c1 = corners1[0, 0].astype(np.int32)
            p0_c2 = corners2[0, 0].astype(np.int32)

            cv.putText(
                frame0,
                "O",
                (p0_c1[0], p0_c1[1]),
                cv.FONT_HERSHEY_DUPLEX,
                1,
                (0, 0, 255),
                1,
            )
            cv.drawChessboardCorners(frame0, (rows, cols), corners1, c_ret1)
            cv.imshow("img", frame0)

            cv.putText(
                frame1,
                "O",
                (p0_c2[0], p0_c2[1]),
                cv.FONT_HERSHEY_DUPLEX,
                1,
                (0, 0, 255),
                1,
            )
            cv.drawChessboardCorners(frame1, (rows, cols), corners2, c_ret2)
            cv.imshow("img2", frame1)
            k = cv.waitKey(0)

            if k & 0xFF == ord("s"):
                print("skipping")
                continue

            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
            objpoints.append(objp)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        mtx0,
        dist0,
        mtx1,
        dist1,
        (width, height),
        criteria=criteria,
        flags=stereocalibration_flags,
    )

    print("rmse: ", ret)
    print("CM1: ", CM1)
    print("CM2: ", CM2)
    print("dist0: ", dist0)
    print("dist1: ", dist1)
    print(": R", R)
    print(": T", T)
    print(": E", E)
    print(": F", F)
    cv.destroyAllWindows()
    return R, T


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
