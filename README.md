# VRChat OSC Tracker using MediaPipe

Uses [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) for human pose estimation and sends hip and feet position and rotation to [VRChat using OSC](https://docs.vrchat.com/docs/osc-trackers).

## Usage

```
python main.py [-h] [--frames | --intrinsic | --extrinsic | --transform] [--config]

VRChat OSC Tracker. Run with no arguments to start the tracking.

options:
  -h, --help   show this help message and exit
  --frames     Take frames with both cameras and save them.
  --intrinsic  Obtain intrinsic parameters for each camera and save them.
  --extrinsic  Obtain extrinsic parameters.
  --transform  Obtain rotation and translation for each camera.
  --config     Config file to load.
```
