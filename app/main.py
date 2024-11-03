import argparse

import track
import calibration


def main():
    parser = argparse.ArgumentParser(
        description="VRChat OSC Tracker. Run with no arguments to start the tracking."
    )
    group = parser.add_mutually_exclusive_group(required=False)

    group.add_argument(
        "--frames",
        action="store_true",
        help="Take frames with both cameras and save them.",
    )
    group.add_argument(
        "--intrinsic",
        action="store_true",
        help="Obtain and save intrinsic parameters for each camera.",
    )
    group.add_argument(
        "--stereo",
        action="store_true",
        help="Stereo calibrate and save the extrinsic parameters for each camera.",
    )

    parser.add_argument(
        "--config",
        action="store_true",
        help="Config file to load.",
        default="calibration_settings.yaml",
    )

    args = parser.parse_args()

    calibration.parse_settings(args.config)

    if args.frames:
        calibration.frames()
    elif args.intrinsic:
        calibration.intrinsic()
    elif args.stereo:
        calibration.stereo_calibrate()
    else:
        track.start()


if __name__ == "__main__":
    main()
