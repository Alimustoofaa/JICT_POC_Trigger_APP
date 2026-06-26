import os
import cv2
import numpy as np
from pathlib import Path

try: import yaml
except ImportError: yaml = None
try: import supervision as sv
except ImportError: sv = None

try: from src.detection import ConfigModel as ConfigModelDetection, \
		Detection, InputDetection, PostProcessing
except ImportError:
	try: from detection import ConfigModel as ConfigModelDetection, \
		Detection, InputDetection, PostProcessing
	except ImportError:
		ConfigModelDetection = Detection = InputDetection = PostProcessing = None

try: from src.helper import load_polygon_config, initializetion_camera
except: from helper import load_polygon_config, initializetion_camera
 
try: from src.utils import get_parent_directory, is_rtsp_or_local_file, \
	calculate_bbox_original, image_letterbox, filter_bboxes_in_polygon, \
	draw_polygon, draw_polygons, draw_results, draw_detection, draw_tracking
except ImportError:
	try: from utils import get_parent_directory, is_rtsp_or_local_file, \
		calculate_bbox_original, image_letterbox, filter_bboxes_in_polygon, \
		draw_polygon, draw_polygons, draw_results, draw_detection, draw_tracking
	except ImportError:
		get_parent_directory = is_rtsp_or_local_file = None
		calculate_bbox_original = image_letterbox = None
		filter_bboxes_in_polygon = draw_polygon = draw_polygons = draw_results = draw_detection = draw_tracking = None
	
class DetectionTrigger:
	def __init__(
			self,
			camera_config_path: str,
			model_config_path: str,
			virtual_lane_config_path: str
		):
		self.camera_config_path = camera_config_path
		self.model_config_path = model_config_path
		self.virtual_lane_config_path = virtual_lane_config_path
		self.app_isrunning = True
		self.polygon_points = load_polygon_config(self.virtual_lane_config_path)
		self.cameras = self.initializetion_camera()
		self.camera_streams = {}
		self.model_config = self.__load_model_config()
		self.devices = self.__get_inference_devices()
		self.__load_model()
		self.tracker = self.__create_tracker()
		print(f"Loaded polygon points: {self.polygon_points}")
		print(f"Initialized cameras: {list(self.cameras.keys())}")
		print(f"Tracking with supervision: {self.tracker is not None}")

	def initializetion_camera(self) -> dict:
		return initializetion_camera(self.camera_config_path)

	def __load_model_config(self) -> dict:
		if not yaml:
			return {}

		config_path = Path(self.model_config_path)
		if not config_path.is_file():
			alternate_path = config_path.with_name("models.yaml")
			if alternate_path.is_file():
				config_path = alternate_path
			else:
				return {}

		with config_path.open("r", encoding="utf-8") as file:
			return yaml.safe_load(file) or {}
		
	########################################################
	# Load Detection Model
	########################################################
	def __load_detection(self, config: dict):
		if not all((ConfigModelDetection, Detection, InputDetection, PostProcessing)):
			raise RuntimeError("Detection dependencies are not available")

		detection_config = ConfigModelDetection(
			name = config['name'],
			version = config['version'],
			model_path = os.path.join(get_parent_directory(), config['model_path']),
			model_format = config['model_format'],
			input_size = config['input_size'],
			classes = config['classes'],
			filter_classes = config['filter_classes'],
			postprocessing = PostProcessing(
				confidence_threshold = config['postprocessing']['confidence_threshold'],
				nms_threshold = config['postprocessing']['nms_threshold']
			),
			devices = self.devices,
		)
		detection = Detection(config=detection_config)
		return detection

	def __create_tracker(self):
		if sv is None:
			return None
		return sv.ByteTrack()
  
	def __get_inference_devices(self):
		return self.model_config.get('inference', {}).get('devices', ["CPUExecutionProvider"])
  
	def __load_model(self):
		if not self.model_config:
			self.model_detection_trigger = None
			self.model_yolo = None
			self.time_skiped_process = 0.0
			return

		self.model_detection_trigger = self.__load_detection(self.model_config['trigger_detection'])
		self.time_skiped_process = self.model_config['trigger_detection'].get('skiped_process', 0.0)
	
		try: self.model_yolo = self.__load_detection(self.model_config['yolo_detection'])
		except Exception as e: self.model_yolo = None
  
	def detection(
	 	self, 
	  	image: np.ndarray, 
	   	model: Detection, 
		is_letterbox: bool = True, 
		filter_classes: list = None
	) -> tuple[list, list, list]:
		try:
			if is_letterbox:
				input_image, ratio, dwdh = image_letterbox(
					image, 
					new_shape=(model.input_size[0], model.input_size[1]),
					color=(0, 0, 0)
				)
				dwdh = np.array(dwdh * 2, dtype=np.float32)
			else: input_image = image
				
			bboxs, scores, labels = model(
					InputDetection(
						image = input_image,
						filter_classes = filter_classes
					)
				)
			if is_letterbox:
				bboxs = [
						calculate_bbox_original(
							bbox=i, dwdh=dwdh, ratio=ratio
						) for i in bboxs
					]
			else: bboxs = [list(map(int, bbox)) for bbox in bboxs]
			scores = [float(round(score, 2)) for score in scores]
		except:
			bboxs, scores, labels = [], [], []
		return bboxs, scores, labels

	def track_objects(
		self,
		bboxs: list,
		scores: list,
		labels: list,
	) -> tuple[list, list, list, list]:
		if not bboxs:
			return [], [], [], []

		if self.tracker is None or sv is None:
			return bboxs, scores, labels, list(range(1, len(bboxs) + 1))

		detections = sv.Detections(
			xyxy=np.array(bboxs, dtype=np.float32),
			confidence=np.array(scores, dtype=np.float32),
			data={"class_name": np.array(labels, dtype=object)},
		)
		tracked = self.tracker.update_with_detections(detections)

		tracked_bboxs = tracked.xyxy.astype(int).tolist()
		tracked_scores = (
			tracked.confidence.tolist()
			if tracked.confidence is not None
			else [0.0] * len(tracked_bboxs)
		)
		tracked_labels = list(tracked.data.get("class_name", []))
		tracker_ids = (
			tracked.tracker_id.astype(int).tolist()
			if tracked.tracker_id is not None
			else list(range(1, len(tracked_bboxs) + 1))
		)

		return tracked_bboxs, tracked_scores, tracked_labels, tracker_ids

	def filter_detections_in_polygons(
		self,
		bboxs: list,
		scores: list,
		labels: list,
		tracker_ids: list,
	) -> tuple[list, list, list, list]:
		filtered_bboxs = []
		filtered_scores = []
		filtered_labels = []
		filtered_tracker_ids = []

		for lane_name, polygon_points in self.polygon_points.items():
			lane_bboxs, lane_scores, lane_labels = filter_bboxes_in_polygon(
				bboxs,
				scores,
				labels,
				polygon_points,
			)

			for lane_bbox, lane_score, lane_label in zip(
				lane_bboxs,
				lane_scores,
				lane_labels,
			):
				try:
					index = next(
						i for i, bbox in enumerate(bboxs)
						if list(map(int, bbox)) == list(map(int, lane_bbox))
					)
				except StopIteration:
					continue

				if tracker_ids:
					tracker_id = tracker_ids[index]
				else:
					tracker_id = index + 1

				print(
					f"{lane_name} | object={lane_label} | confidence={lane_score:.2f}"
				)

				filtered_bboxs.append(lane_bbox)
				filtered_scores.append(lane_score)
				filtered_labels.append(lane_label)
				filtered_tracker_ids.append(tracker_id)

		return filtered_bboxs, filtered_scores, filtered_labels, filtered_tracker_ids

	def open_camera(self, camera_id: str):
		camera = self.cameras.get(camera_id)
		if camera is None:
			raise ValueError(f"Camera '{camera_id}' not found")

		stream_url = camera.get_rtsp_url()
		print(f"{camera_id} camera RTSP URL:")
		print(stream_url)

		cap = cv2.VideoCapture(stream_url)
		if not cap.isOpened():
			raise RuntimeError(f"Failed to open camera stream: {stream_url}")

		self.camera_streams[camera_id] = cap
		return cap

	def close_camera(self, camera_id: str):
		cap = self.camera_streams.pop(camera_id, None)
		if cap is not None:
			cap.release()
		cv2.destroyAllWindows()
  
		
	def trigger_process(
		self,
		model: Detection,
		camera_id: str,
		filter_classes: list,
	):
		camera = self.cameras.get(camera_id)
		cap = self.open_camera(camera_id)

		try:
			while self.app_isrunning:
				ret, frame = cap.read()
				if type(frame) is not np.ndarray or frame is None:
					continue
				bboxs, scores, labels = self.detection(
					image=frame,
					model=model,
					is_letterbox=True,
					filter_classes=filter_classes
				)
				tracked_bboxs, tracked_scores, tracked_labels, tracker_ids = self.track_objects(
					bboxs=bboxs,
					scores=scores,
					labels=labels,
				)
				filtered_bboxs, filtered_scores, filtered_labels, filtered_tracker_ids = self.filter_detections_in_polygons(
					bboxs=tracked_bboxs,
					scores=tracked_scores,
					labels=tracked_labels,
					tracker_ids=tracker_ids,
				)

				if filtered_bboxs:
					frame = draw_tracking(
						frame,
						filtered_bboxs,
						filtered_scores,
						filtered_labels,
						filtered_tracker_ids,
					)
				else:
					frame = draw_detection(frame, filtered_bboxs, filtered_scores, filtered_labels)
				frame = draw_polygons(frame, self.polygon_points)
				cv2.imshow(f"Camera {camera_id}", frame)

				if cv2.waitKey(1) & 0xFF == ord("q"):
					break
		finally:
			self.close_camera(camera_id)

		return camera
		

	def run(self):
		self.trigger_process(
			model=self.model_detection_trigger,
			camera_id="trigger",
			filter_classes=None,
		)

		return None
		
		
if __name__ == "__main__":
	# =========================================================
	# CONFIG
	# =========================================================
	CAMERA_CFG = "config/cameras.json"
	LANE_CFG = "config/lane.json"

	MODEL_CFG = "config/models.yaml"
	SERVICE_CFG = "config/service_config.yaml"

	# =========================================================
	# ENGINE INIT
	# =========================================================
	engine = DetectionTrigger(
		camera_config_path=CAMERA_CFG,
		model_config_path=MODEL_CFG,
		virtual_lane_config_path=LANE_CFG,
	)
	engine.run()
