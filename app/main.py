import argparse

import track
import calibration


def get_intrinsic_parameters():
    print("Obtaining intrinsic parameters for each camera and saving them.")


def get_extrinsic_parameters():
    print("Obtaining extrinsic parameters and saving them.")


def get_transformation():
    print("Obtaining rotation and translation for each camera.")


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
        help="Obtain intrinsic parameters for each camera and save them.",
    )
    group.add_argument(
        "--extrinsic",
        action="store_true",
        help="Obtain extrinsic parameters.",
    )
    group.add_argument(
        "--transform",
        action="store_true",
        help="Obtain rotation and translation for each camera.",
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
        get_intrinsic_parameters()
    elif args.extrinsic:
        get_extrinsic_parameters()
    elif args.transform:
        get_transformation()
    else:
        track.start()


if __name__ == "__main__":
    main()
