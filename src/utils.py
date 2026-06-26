from fcntl import fcntl
import os
import struct
import sys
import cv2
import socket
import base64
from typing import List, Union, Tuple
from numpy import ndarray, int32, float32, array
from shapely.geometry import Point, Polygon




def get_parent_directory():
	"""
	Get the parent directory of the current script or PyInstaller executable.
	"""
	if getattr(sys, 'frozen', False):
		base_path = os.path.dirname(sys.executable)
	else:
		base_path = os.path.dirname(os.path.abspath(__file__))
	return os.path.dirname(base_path)

def is_rtsp_or_local_file(source: str):
	"""
	Check if the source is an RTSP stream or a local file path.
	"""
	if not source:
		raise ValueError("Source cannot be empty")
	
	if source.startswith("rtsp://") and "://" in source:
		return True
	
	if os.path.isfile(source):
		return False
	
def image_letterbox(
		im: ndarray,
		new_shape: Union[Tuple, List] = (1024, 1024),
		color: Union[Tuple, List] = (255, 255, 255)
	) -> Tuple[array, float, Tuple[float, float]]:
	# Resize and pad image while meeting stride-multiple constraints
	shape = im.shape[:2]  # current shape [height, width]
	if isinstance(new_shape, int):
		new_shape = (new_shape, new_shape)

	r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # Scale ratio
	new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # new width, height
	dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
	dw /= 2
	dh /= 2

	if shape[::-1] != new_unpad:
		im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

	top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
	left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
	im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
	return im, r, (dw, dh)

def calculate_bbox_original(bbox, dwdh, ratio):
	bbox = array(bbox, dtype=float32)
	bbox = (bbox - dwdh) / ratio
	return bbox.astype(int32)

def adjust_bbox(bbox, img_width, img_height):
	xmin, ymin, xmax, ymax = bbox
	xmin = max(0, xmin)
	ymin = max(0, ymin)
	xmax = min(img_width, xmax)
	ymax = min(img_height, ymax)
	return array([xmin, ymin, xmax, ymax], dtype=int32)

def filter_bboxes_in_polygon(bboxes, scores, labels, polygon_points):
	"""
	Filters bounding boxes whose center point is inside the polygon (using shapely).
	"""
	# Handle deeply nested input like [[[ [x, y], [x2, y2], ... ]]]
	while isinstance(polygon_points[0], (list, tuple)) and not isinstance(polygon_points[0][0], (int, float)):
		polygon_points = polygon_points[0]

	# Convert to list of (x, y) tuples
	polygon_coords = [tuple(map(float, p)) for p in polygon_points]
	polygon = Polygon(polygon_coords)

	filtered_bboxes = []
	filtered_scores = []
	filtered_labels = []

	for bbox, score, label in zip(bboxes, scores, labels):
		x1, y1, x2, y2 = bbox
		cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
		center = Point(cx, cy)

		if polygon.contains(center):
			filtered_bboxes.append(bbox)
			filtered_scores.append(score)
			filtered_labels.append(label)

	return filtered_bboxes, filtered_scores, filtered_labels

def normalize_polygon(pts, frame):
    """
    pts : [[x,y], ...] pixel
    return : normalized polygon [[0-1]]
    """
    h, w = frame.shape[:2]
    return [[x / w, y / h] for x, y in pts]

def denormalize_polygon(pts_norm, frame):
    """
    pts_norm : [[0-1]]
    return : pixel polygon
    """
    h, w = frame.shape[:2]
    return [[int(x * w), int(y * h)] for x, y in pts_norm]

def draw_polygon(image, polygon_points, color=(255, 153, 31), thickness=3):
	"""
	Draws a polygon on the image.

	Args:
		image (np.ndarray): The input image
		polygon_points (List[Tuple[int, int]]): Points of the polygon
		color (Tuple[int, int, int]): BGR color (default orange)
		thickness (int): Line thickness (default 2)
	"""
	img = image.copy()
	pts = array(polygon_points, int32).reshape((-1, 1, 2))
	cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

	return img

def draw_polygons(image, polygons, color=(255, 153, 31), thickness=3, show_label=True):
	"""
	Draw multiple labeled polygons on the image.

	Args:
		image (np.ndarray): The input image
		polygons (Dict[str, List[List[int | float]]]): Labeled polygon points
		color (Tuple[int, int, int]): BGR color
		thickness (int): Line thickness
		show_label (bool): Draw polygon label text near the first point
	"""
	img = image.copy()

	for label, polygon_points in polygons.items():
		pts = array(polygon_points, int32).reshape((-1, 1, 2))
		cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

		if show_label:
			text_x, text_y = pts[0][0]
			cv2.putText(
				img,
				str(label),
				(int(text_x), int(text_y) - 10),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				color,
				2,
				cv2.LINE_AA
			)

	return img

def draw_results(image, bboxs, scores, labels):
    """
    Simple visualization:
    - Light blue bounding box
    - Light blue label background
    - White text
    """

    img = image.copy()

    # Light blue color (BGR)
    color_blue_light = (255, 204, 102)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    for bbox, score, label in zip(bboxs, scores, labels):
        x_min, y_min, x_max, y_max = map(int, bbox)

        # Draw bounding box
        cv2.rectangle(
            img,
            (x_min, y_min),
            (x_max, y_max),
            color_blue_light,
            2
        )

        # Label text
        text = f"{label}: {int(score * 100)}%"

        (tw, th), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Draw label background (single line)
        cv2.rectangle(
            img,
            (x_min, y_min - th - baseline - 4),
            (x_min + tw + 6, y_min),
            color_blue_light,
            -1
        )

        # Draw label text
        cv2.putText(
            img,
            text,
            (x_min + 3, y_min - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return img

def draw_detection(image, bboxs, scores, labels):
    """
    Draw detection results on the image.

    This is a small named wrapper so detection visualization can be reused
    explicitly from application code.
    """
    return draw_results(image, bboxs, scores, labels)

def draw_tracking(image, bboxs, scores, labels, tracker_ids):
    """
    Draw tracked detections with tracker ids on the image.
    """
    img = image.copy()
    color_green = (80, 200, 120)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    for bbox, score, label, tracker_id in zip(bboxs, scores, labels, tracker_ids):
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color_green, 2)

        text = f"#{tracker_id} {label}: {int(score * 100)}%"
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(
            img,
            (x_min, y_min - th - baseline - 6),
            (x_min + tw + 8, y_min),
            color_green,
            -1,
        )
        cv2.putText(
            img,
            text,
            (x_min + 4, y_min - baseline - 3),
            font,
            font_scale,
            (20, 20, 20),
            thickness,
            cv2.LINE_AA,
        )

    return img

def draw_trigger_center(frame, text="TRIGGER"):
    h, w = frame.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3.0
    thickness = 6 
    color = (0, 0, 255)

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Posisi tengah frame
    x = (w - tw) // 2
    y = (h + th) // 2

    cv2.putText(
        frame, text,
        (x + 3, y + 3),
        font, font_scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame, text,
        (x, y),
        font, font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )

    return frame
