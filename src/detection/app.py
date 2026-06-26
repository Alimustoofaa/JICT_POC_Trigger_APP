"""
@Author     : Ali Mustofa
@Module     : Interface Detection
@Email      : hai.alimustofa@gmail.com
@Created on : 16 Desember 2025
"""

from __future__ import annotations

import os
import sys
from pathlib import PosixPath
from typing import Optional

import numpy as np
import onnxruntime
from cv2 import COLOR_BGR2RGB, cvtColor, resize
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
from pydantic import BaseModel


class InputDetection(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    image: np.ndarray
    filter_classes: Optional[list[str]] = None
    confidence_threshold: Optional[float] = 0.0
    nms_threshold: Optional[float] = 0.0


class PostProcessing(BaseModel):
    confidence_threshold: Optional[float] = 0.5
    nms_threshold: Optional[float] = 0.4


class ConfigModel(BaseModel):
    name: str
    version: str
    model_path: PosixPath
    model_format: str
    input_size: list[int]
    classes: list[str]
    filter_classes: Optional[list[str]]
    postprocessing: PostProcessing
    devices: Optional[list] = ["CPUExecutionProvider"]


class Detection:
    def __init__(self, config: ConfigModel):
        """
        Load model.
        @params:
            - config_name:str -> Config name for detection
        """
        self.config = config
        self.__get_config()
        self.ort_session: InferenceSession = self.get_ort_session()
        self.classes, self.input_shape = self.get_metamodel()
        self.input_names = self.get_inputmodel()
        self.input_height, self.input_width = self.input_shape
        self.output_names: list = self.get_outputmodel()

    def __get_config(self) -> None:
        self.name = self.config.name
        self.version = self.config.version
        self.model_path = self.config.model_path
        self.model_format = self.config.model_format
        self.input_size = self.config.input_size
        self.classes = self.config.classes
        self.filter_classes = self.config.filter_classes
        self.confidence_threshold = self.config.postprocessing.confidence_threshold
        self.nms_threshold = self.config.postprocessing.nms_threshold
        self.devices = self.config.devices

        config_attributes = {
            "Name": self.name,
            "Version": self.version,
            "Model Path": self.model_path,
            "Model Format": self.model_format,
            "Input Size": self.input_size,
            "Classes": self.classes,
            "Filter Classes": self.filter_classes,
            "Confidence Threshold": self.confidence_threshold,
            "NMS Threshold": self.nms_threshold,
            "Devices": self.devices,
        }

        print("=" * 60)
        print("Model Detection Configuration")
        for key, value in config_attributes.items():
            print(f"- {key} : {value}")
        print("=" * 60)

    def get_ort_session(self) -> InferenceSession:
        model_path_name = (
            self.model_path / f"{self.name}_{self.version}.{self.model_format}"
        )
        if not os.path.isfile(model_path_name):
            sys.exit(f"Model : {model_path_name} Not Found !!!")

        opt_session = onnxruntime.SessionOptions()
        opt_session.enable_mem_pattern = True
        opt_session.enable_cpu_mem_arena = True
        opt_session.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        )

        ort_session = onnxruntime.InferenceSession(
            model_path_name,
            providers=self.devices,
            opt_session=opt_session,
        )
        return ort_session

    def get_metamodel(self) -> tuple:
        model_inputs = self.ort_session.get_modelmeta()
        classes = eval(model_inputs.custom_metadata_map["names"])
        input_shape = eval(model_inputs.custom_metadata_map["imgsz"])
        return classes, input_shape

    def get_inputmodel(self) -> list:
        model_inputs = self.ort_session.get_inputs()
        input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        return input_names

    def get_outputmodel(self) -> list:
        model_output = self.ort_session.get_outputs()
        output_names = [model_output[i].name for i in range(len(model_output))]
        return output_names

    def image2tensor(self, image: np.ndarray) -> np.ndarray:
        image_rgb = cvtColor(image, COLOR_BGR2RGB)
        resized = resize(image_rgb, (self.input_width, self.input_height))
        input_image = resized / 255.0
        input_image = input_image.transpose(2, 0, 1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def __rescale_boxes(self, boxes: np.ndarray, image_shape: tuple) -> np.ndarray:
        image_height, image_width = image_shape[:2]
        input_shape = np.array(
            [
                self.input_width,
                self.input_height,
                self.input_width,
                self.input_height,
            ]
        )
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_width, image_height, image_width, image_height])
        boxes = boxes.astype(np.int32)
        return boxes

    @staticmethod
    def nms(boxes, scores, iou_threshold, compute_iou):
        sorted_indices = np.argsort(scores)[::-1]
        keep_boxes = []
        while sorted_indices.size > 0:
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)
            ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
            keep_indices = np.where(ious < iou_threshold)[0]
            sorted_indices = sorted_indices[keep_indices + 1]
        return keep_boxes

    @staticmethod
    def compute_iou(box, boxes):
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area
        iou = intersection_area / union_area
        return iou

    @staticmethod
    def __xywh2xyxy(x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def __extract_output(
        self,
        outputs: list,
        thresold_val: float,
        nms_val: float,
        image_shape: tuple,
    ):
        predictions = np.squeeze(outputs[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > thresold_val, :]
        scores = scores[scores > thresold_val]
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = predictions[:, :4]
        boxes = self.__rescale_boxes(boxes, image_shape)
        boxes = self.__xywh2xyxy(boxes)
        indices = self.nms(boxes, scores, nms_val, self.compute_iou)
        class_str = [self.classes[i] for i in class_ids[indices].tolist()]
        return boxes[indices], scores[indices], np.array(class_str)

    def filter_result(self, results, filter_classes):
        boxes, scores, classes = results
        filtered = [item in filter_classes for item in classes]
        return boxes[filtered], scores[filtered], classes[filtered]

    def __call__(self, input: InputDetection):
        image = input.image
        image_shape = image.shape
        confidence_threshold = (
            input.confidence_threshold
            if input.confidence_threshold
            else self.confidence_threshold
        )
        nms_threshold = input.nms_threshold if input.nms_threshold else self.nms_threshold
        filter_classes = (
            input.filter_classes if input.filter_classes else self.filter_classes
        )

        input_tensor = self.image2tensor(image)
        outputs = self.ort_session.run(
            self.output_names,
            {self.input_names[0]: input_tensor},
        )
        results = self.__extract_output(
            outputs=outputs,
            thresold_val=confidence_threshold,
            nms_val=nms_threshold,
            image_shape=image_shape,
        )
        if filter_classes:
            results = self.filter_result(results, filter_classes)
        return results
