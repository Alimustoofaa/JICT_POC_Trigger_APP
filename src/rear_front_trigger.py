import os
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

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

try: from src.camera.hikvision.stream import FFmpegRTSPCamera
except ImportError:
	try: from camera.hikvision.stream import FFmpegRTSPCamera
	except ImportError: FFmpegRTSPCamera = None
 
try: from src.utils import get_parent_directory, is_rtsp_or_local_file, \
	adjust_bbox, calculate_bbox_original, image_letterbox, filter_bboxes_in_polygon, \
	draw_polygon, draw_polygons, draw_results, draw_detection, draw_tracking
except ImportError:
	try: from utils import get_parent_directory, is_rtsp_or_local_file, \
		adjust_bbox, calculate_bbox_original, image_letterbox, filter_bboxes_in_polygon, \
		draw_polygon, draw_polygons, draw_results, draw_detection, draw_tracking
	except ImportError:
		get_parent_directory = is_rtsp_or_local_file = None
		adjust_bbox = calculate_bbox_original = image_letterbox = None
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
		self.front_camera_ids = ["front_1", "front_2", "front_3", "front_4"]
		self.rear_camera_ids = ["rear_4", "rear_3", "rear_2", "rear_1"]
		self.polygon_points = load_polygon_config(self.virtual_lane_config_path)
		self.cameras = self.initializetion_camera()
		self.camera_streams = {}
		self.max_failed_reads = 30
		self.reconnect_delay = 1.0
		self.capture_dir = Path("captures/app")
		self.video_output_dir = Path("outputs/video")
		self.video_writers = {}
		self.model_config = self.__load_model_config()
		self.devices = self.__get_inference_devices()
		self.__load_model()
		print(f"Loaded polygon points: {self.polygon_points}")
		print(f"Initialized cameras: {list(self.cameras.keys())}")
		print(f"Tracking with supervision: {sv is not None}")

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

	def get_tracker(self, tracker_key: str):
		tracker = getattr(self, "trackers", {}).get(tracker_key)
		if tracker_key not in getattr(self, "trackers", {}):
			if not hasattr(self, "trackers"):
				self.trackers = {}
			self.trackers[tracker_key] = self.__create_tracker()
			tracker = self.trackers[tracker_key]
		return tracker
  
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
		if image is None or model is None:
			return [], [], []

		try:
			frame_height, frame_width = image.shape[:2]

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
				original_bboxs = []
				for bbox in bboxs:
					original_bbox = calculate_bbox_original(
						bbox=bbox,
						dwdh=dwdh,
						ratio=ratio,
					)
					original_bbox = adjust_bbox(
						original_bbox,
						frame_width,
						frame_height,
					)
					original_bboxs.append(original_bbox.astype(int).tolist())
				bboxs = original_bboxs
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
		tracker_key: str,
	) -> tuple[list, list, list, list]:
		if not bboxs:
			return [], [], [], []

		tracker = self.get_tracker(tracker_key)
		if tracker is None or sv is None:
			return bboxs, scores, labels, list(range(1, len(bboxs) + 1))

		detections = sv.Detections(
			xyxy=np.array(bboxs, dtype=np.float32),
			confidence=np.array(scores, dtype=np.float32),
			data={"class_name": np.array(labels, dtype=object)},
		)
		tracked = tracker.update_with_detections(detections)

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
	) -> tuple[list, list, list, list, list]:
		filtered_bboxs = []
		filtered_scores = []
		filtered_labels = []
		filtered_tracker_ids = []
		filtered_lane_names = []

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
				filtered_lane_names.append(lane_name)

		return filtered_bboxs, filtered_scores, filtered_labels, filtered_tracker_ids, filtered_lane_names

	def open_camera(self, camera_id: str):
		camera = self.cameras.get(camera_id)
		if camera is None:
			raise ValueError(f"Camera '{camera_id}' not found")

		stream_url = camera.get_rtsp_url()
		print(f"{camera_id} camera RTSP URL:")
		print(stream_url)

		use_ffmpeg = stream_url.startswith("rtsp://") and FFmpegRTSPCamera is not None

		if use_ffmpeg:
			cap = FFmpegRTSPCamera(
				rtsp_url=stream_url,
				camera_id=camera_id,
				width=int(camera.config.get("width", 1920)),
				height=int(camera.config.get("height", 1080)),
				fps=int(camera.config.get("fps", 15)),
				read_timeout=int(camera.config.get("read_timeout", 10)),
			)
			cap.open()
		else:
			cap = cv2.VideoCapture(stream_url)
			cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
			if not cap.isOpened():
				raise RuntimeError(f"Failed to open camera stream: {stream_url}")

		self.camera_streams[camera_id] = cap
		return cap

	def close_camera(self, camera_id: str):
		cap = self.camera_streams.pop(camera_id, None)
		if cap is not None:
			if hasattr(cap, "release"):
				cap.release()
			elif hasattr(cap, "close"):
				cap.close()
		cv2.destroyAllWindows()

	def read_camera_frame(self, camera_id: str):
		cap = self.camera_streams.get(camera_id)
		if cap is None:
			return False, None

		if hasattr(cap, "read") and hasattr(cap, "close") and not hasattr(cap, "release"):
			frame = cap.read()
			return frame is not None, frame

		ret, frame = cap.read()
		return ret, frame

	def open_cameras(self, camera_ids: list[str]):
		for camera_id in camera_ids:
			if camera_id not in self.camera_streams:
				self.open_camera(camera_id)

	def close_cameras(self, camera_ids: list[str]):
		for camera_id in camera_ids:
			self.close_camera(camera_id)

	def merge_camera_frames(
		self,
		frames: dict[str, np.ndarray],
		camera_ids: list[str],
	) -> np.ndarray | None:
		if not frames:
			return None

		base_frame = next(iter(frames.values()))
		frame_height, frame_width = base_frame.shape[:2]
		merged_frames = []

		for camera_id in camera_ids:
			frame = frames.get(camera_id)
			if frame is None:
				frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
			merged_frames.append(frame)

		return np.hstack(merged_frames)

	def read_merged_frame(self, camera_ids: list[str]):
		frames = {}
		failed_camera_ids = []

		for camera_id in camera_ids:
			ret, frame = self.read_camera_frame(camera_id)
			if not ret or frame is None or not isinstance(frame, np.ndarray):
				failed_camera_ids.append(camera_id)
				continue
			frames[camera_id] = frame

		if failed_camera_ids:
			return False, None, failed_camera_ids

		return True, self.merge_camera_frames(frames, camera_ids), []

	def read_merged_front_frame(self, camera_ids: list[str]):
		return self.read_merged_frame(camera_ids)

	def reopen_camera(self, camera_id: str):
		print(f"Reconnecting camera: {camera_id}")
		self.close_camera(camera_id)
		time.sleep(self.reconnect_delay)
		return self.open_camera(camera_id)

	def save_capture(self, frame: np.ndarray, camera_id: str) -> Path:
		self.capture_dir.mkdir(parents=True, exist_ok=True)
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
		output_path = self.capture_dir / f"{camera_id}_{timestamp}.jpg"
		cv2.imwrite(str(output_path), frame)
		return output_path

	def open_video_writer(self, frame: np.ndarray, camera_id: str):
		video_writer = self.video_writers.get(camera_id)
		if video_writer is not None:
			return video_writer

		self.video_output_dir.mkdir(parents=True, exist_ok=True)
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		output_path = self.video_output_dir / f"{camera_id}_{timestamp}.mp4"
		frame_height, frame_width = frame.shape[:2]
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		video_writer = cv2.VideoWriter(
			str(output_path),
			fourcc,
			15.0,
			(frame_width, frame_height),
		)
		self.video_writers[camera_id] = video_writer
		print(f"Recording video: {output_path}")
		return video_writer

	def close_video_writer(self, camera_id: str | None = None):
		if camera_id is not None:
			video_writer = self.video_writers.pop(camera_id, None)
			if video_writer is not None:
				video_writer.release()
			return

		for video_writer in self.video_writers.values():
			video_writer.release()
		self.video_writers.clear()

	def handle_failed_reads(
		self,
		active_camera_ids: list[str],
		failed_camera_ids: list[str],
		failed_reads: dict[str, int],
	):
		for camera_id in failed_camera_ids:
			failed_reads[camera_id] += 1
			if failed_reads[camera_id] >= self.max_failed_reads:
				self.reopen_camera(camera_id)
				failed_reads[camera_id] = 0

		successful_camera_ids = set(active_camera_ids) - set(failed_camera_ids)
		for camera_id in successful_camera_ids:
			failed_reads[camera_id] = 0

	def process_frame(
		self,
		frame: np.ndarray,
		model: Detection,
		filter_classes: list,
		tracker_key: str,
	):
		bboxs, scores, labels = self.detection(
			image=frame,
			model=model,
			is_letterbox=True,
			filter_classes=filter_classes,
		)
		tracked_bboxs, tracked_scores, tracked_labels, tracker_ids = self.track_objects(
			bboxs=bboxs,
			scores=scores,
			labels=labels,
			tracker_key=tracker_key,
		)
		filtered_bboxs, filtered_scores, filtered_labels, filtered_tracker_ids, filtered_lane_names = self.filter_detections_in_polygons(
			bboxs=tracked_bboxs,
			scores=tracked_scores,
			labels=tracked_labels,
			tracker_ids=tracker_ids,
		)

		if filtered_bboxs:
			output_frame = draw_tracking(
				frame,
				filtered_bboxs,
				filtered_scores,
				filtered_labels,
				filtered_tracker_ids,
				filtered_lane_names,
			)
		else:
			output_frame = draw_detection(frame, filtered_bboxs, filtered_scores, filtered_labels)

		output_frame = draw_polygons(
			output_frame,
			self.polygon_points,
			active_labels=set(filtered_lane_names),
		)

		return output_frame, filtered_lane_names
  
		
	def trigger_process(
		self,
		model: Detection,
		camera_id: str | list[str],
		filter_classes: list,
		tracker_key: str | None = None,
	):
		is_merged_front = isinstance(camera_id, list)
		active_camera_ids = camera_id if is_merged_front else [camera_id]
		tracker_key = tracker_key or ("front_merge" if is_merged_front else str(camera_id))
		camera = [self.cameras.get(cam_id) for cam_id in active_camera_ids] if is_merged_front else self.cameras.get(camera_id)
		self.open_cameras(active_camera_ids)
		failed_reads = {cam_id: 0 for cam_id in active_camera_ids}

		try:
			while self.app_isrunning:
				if is_merged_front:
					ret, frame, failed_camera_ids = self.read_merged_frame(active_camera_ids)
					self.handle_failed_reads(active_camera_ids, failed_camera_ids, failed_reads)
					if not ret or frame is None:
						continue
				else:
					ret, frame = self.read_camera_frame(camera_id)
					if not ret or type(frame) is not np.ndarray or frame is None:
						self.handle_failed_reads(active_camera_ids, [camera_id], failed_reads)
						continue
					self.handle_failed_reads(active_camera_ids, [], failed_reads)

				frame, filtered_lane_names = self.process_frame(
					frame=frame,
					model=model,
					filter_classes=filter_classes,
					tracker_key=tracker_key,
				)
				video_camera_id = "front_merge" if is_merged_front else camera_id
				video_writer = self.open_video_writer(frame, video_camera_id)
				video_writer.write(frame)
				window_name = "Camera Front Merge" if is_merged_front else f"Camera {camera_id}"
				cv2.imshow(window_name, frame)
				key = cv2.waitKey(1) & 0xFF

				if key == ord("c"):
					output_path = self.save_capture(frame, video_camera_id)
					print(f"Captured: {output_path}")
				elif key == ord("q"):
					break
		finally:
			self.close_video_writer()
			self.close_cameras(active_camera_ids)

		return camera

	def run_parallel_front_rear(self):
		front_stream_id = "front_merge"
		rear_stream_id = "rear_merge"
		all_camera_ids = self.front_camera_ids + self.rear_camera_ids
		failed_reads = {camera_id: 0 for camera_id in all_camera_ids}
		self.open_cameras(all_camera_ids)

		try:
			with ThreadPoolExecutor(max_workers=2) as executor:
				while self.app_isrunning:
					futures = {}
					frames_to_show = {}

					front_ret, front_frame, front_failed_ids = self.read_merged_frame(self.front_camera_ids)
					self.handle_failed_reads(self.front_camera_ids, front_failed_ids, failed_reads)
					if front_ret and front_frame is not None:
						futures[front_stream_id] = executor.submit(
							self.process_frame,
							front_frame,
							self.model_detection_trigger,
							None,
							front_stream_id,
						)

					rear_ret, rear_frame, rear_failed_ids = self.read_merged_frame(self.rear_camera_ids)
					self.handle_failed_reads(self.rear_camera_ids, rear_failed_ids, failed_reads)
					if rear_ret and rear_frame is not None:
						futures[rear_stream_id] = executor.submit(
							self.process_frame,
							rear_frame,
							self.model_detection_trigger,
							None,
							rear_stream_id,
						)

					if not futures:
						continue

					for stream_id, future in futures.items():
						try:
							processed_frame, _ = future.result()
						except Exception as error:
							print(f"Failed to process {stream_id}: {error}")
							continue

						frames_to_show[stream_id] = processed_frame
						video_writer = self.open_video_writer(processed_frame, stream_id)
						video_writer.write(processed_frame)

					if front_stream_id in frames_to_show:
						cv2.imshow("Camera Front Merge", frames_to_show[front_stream_id])
					if rear_stream_id in frames_to_show:
						cv2.imshow("Camera Rear Merge", frames_to_show[rear_stream_id])

					key = cv2.waitKey(1) & 0xFF
					if key == ord("c"):
						for stream_id, frame in frames_to_show.items():
							output_path = self.save_capture(frame, stream_id)
							print(f"Captured: {output_path}")
					elif key == ord("q"):
						break
		finally:
			self.close_video_writer()
			self.close_cameras(all_camera_ids)
		

	def run(self):
		self.run_parallel_front_rear()

		return None
		
		
if __name__ == "__main__":
	# =========================================================
	# CONFIG
	# =========================================================
	CAMERA_CFG = "config/cameras.json"
	LANE_CFG = "config/front_merge.json"

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
