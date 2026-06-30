# POC_CRANE

POC for camera-based crane lane monitoring with:

- RTSP camera streaming
- polygon lane visualization from JSON config
- ONNX object detection
- optional object tracking with `supervision`
- filtering detections by checking whether the bounding-box centroid is inside a configured polygon

## Structure

- `src/app.py`: main runtime pipeline
- `src/helper.py`: config loaders and camera initialization helpers
- `src/utils.py`: polygon, drawing, and bbox utility functions
- `src/detection/`: ONNX detection wrapper
- `src/camera/`: camera abstractions for Axis and Hikvision
- `config/cameras.json`: camera definitions
- `config/lane.json`: polygon lane config
- `config/models.yaml`: detection model config
- `models/`: ONNX model files

## Requirements

Install the base dependencies:

```bash
pip install -r requirements.txt
```

Current required packages from `requirements.txt`:

- `numpy==1.26.4`
- `opencv-python==4.10.0.84`
- `onnxruntime==1.19.2`
- `pydantic==1.10.18`
- `shapely==2.1.2`

Optional:

- `PyYAML` for loading `config/models.yaml`
- `supervision` for ByteTrack-based tracking

Example:

```bash
pip install pyyaml supervision
```

## Configuration

### Camera config

Camera definitions are stored in [config/cameras.json](/home/ali/Halotec/Source_Code/JICT/DEV/POC_CRANE/config/cameras.json:1).

Each camera includes fields like:

- `brand`
- `ip`
- `username`
- `password`
- `stream_endpoint`
- `capture_endpoint`

### Polygon config

Lane polygons are stored in [config/lane.json](/home/ali/Halotec/Source_Code/JICT/DEV/POC_CRANE/config/lane.json:1) using a LabelMe-style `shapes` array.

### Model config

Detection settings are stored in [config/models.yaml](/home/ali/Halotec/Source_Code/JICT/DEV/POC_CRANE/config/models.yaml:1).

Current trigger model:

- model name: `model_truck_container`
- version: `1.0`
- format: `onnx`
- input size: `640 x 640`

## Features

The current pipeline in [src/app.py](/home/ali/Halotec/Source_Code/JICT/DEV/POC_CRANE/src/app.py:1):

- initializes all configured cameras
- opens the configured camera stream
- loads polygon lanes from `config/lane.json`
- runs detection on each frame
- optionally tracks detections with `supervision.ByteTrack`
- filters detections whose bbox centroid is inside a polygon
- prints lane name, object label, and confidence
- draws:
  - detection/tracking boxes
  - tracking IDs
  - configured polygons

Example printed output:

```text
lane_1 | object=container | confidence=0.92
```

## Trigger Front Process

Front trigger process concept:

1. Stream camera `front_1`, `front_2`, `front_3`, `front_4`
2. Merge camera frames into one output frame
3. Run object detection for `container`
4. Check whether detected object is inside the configured virtual line / polygon
5. Determine the lane result for camera `1 / 2 / 3 / 4`
6. When object is inside the virtual line trigger area, capture image from the process output / camera view

Current related outputs in the project:

- live process runtime: [src/app.py](/home/ali/Halotec/Source_Code/JICT/DEV/POC_CRANE/src/app.py:1)
- front image collector: [get_front_image.py](/home/ali/Halotec/Source_Code/JICT/DEV/POC_CRANE/get_front_image.py:1)
- saved image capture: `captures/app/` or `captures/front/`
- saved output video: `outputs/video/`

## Trigger Rear Process

Rear trigger process concept:

1. Stream camera `rear_1`, `rear_2`, `rear_3`, `rear_4`
2. Merge camera frames into one output frame
3. Run object detection for `container`
4. Check whether detected object is inside the configured virtual line / polygon
5. Determine the lane result for camera `1 / 2 / 3 / 4`
6. When object is inside the virtual line trigger area, capture image from the process output / camera view

Current related outputs in the project:

- rear image collector: [get_rear_image.py](/home/ali/Halotec/Source_Code/JICT/DEV/POC_CRANE/get_rear_image.py:1)
- saved image capture: `captures/rear/`

## Run

Start the application from the project root:

```bash
python3 src/app.py
```

Press `q` to close the OpenCV window.

## Camera usage

The camera layer provides a shared interface for:

- RTSP stream URL
- HTTP stream URL
- snapshot capture
- brand-specific PTZ or zoom controls

See [src/camera/README.md](/home/ali/Halotec/Source_Code/JICT/DEV/POC_CRANE/src/camera/README.md:1) for examples.

## Notes

- If `supervision` is not installed, tracking falls back to simple sequential IDs.
- If `PyYAML` is not installed, model config loading is skipped.
- The app expects reachable RTSP camera endpoints from the runtime environment.
