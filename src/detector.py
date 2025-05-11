# -*- coding: utf-8 -*-
"""
YOLO detector module for defect inspection system.
Supports YOLOv11 models with TensorRT acceleration on Jetson.
"""

import os
import cv2
import time
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import shutil

# Set up logging
logger = logging.getLogger("DefectInspection.Detector")

class YOLODetector:
    """YOLO object detector with support for YOLOv11 models and Jetson DLA acceleration"""
    
    def __init__(self, model_name: str = "yolo11n", 
             conf_thresh: float = 0.25, 
             nms_thresh: float = 0.45,
             models_dir: str = "../models"):
        """
        Initialize YOLO detector
        
        Args:
            model_name: Base name of the model (e.g. "yolov11n", "yolov11s", "yolov11m", "yolov11l")
            conf_thresh: Confidence threshold for detections
            nms_thresh: NMS IoU threshold
            models_dir: Directory to store and load models
        """
        self.model_name = model_name
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.models_dir = models_dir
        self.model = None
        self.class_names = ["deformation"]
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load model
        self._load_model()
        logger.info(f"YOLO detector initialized with model: {self.model_name}")

    def _load_model(self):
        """Load PyTorch model directly without TensorRT conversion"""
        try:
            # Import YOLO from ultralytics
            from ultralytics import YOLO
            
            # Check for PyTorch model path
            pt_path = os.path.join(self.models_dir, f"{self.model_name}.pt")
            
            # Case 1: PyTorch model exists, load directly
            if os.path.exists(pt_path):
                logger.info(f"Loading PyTorch model: {pt_path}")
                self.model = YOLO(pt_path)
                
            # Case 2: Download from Ultralytics
            else:
                logger.info(f"No local model found. Downloading {self.model_name} from Ultralytics")
                
                # Download model
                self.model = YOLO(self.model_name)
                
                # Save the PyTorch model for future use
                self.model.save(pt_path)
                logger.info(f"Saved PyTorch model to {pt_path}")
            
            # Set model parameters
            self.model.conf = self.conf_thresh  # Detection confidence threshold
            self.model.iou = self.nms_thresh    # NMS IoU threshold
            
            # Run a warmup inference
            logger.info("Running warmup inference")
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_input)
            logger.info("PyTorch model loaded and ready for inference")
            
        except ImportError:
            logger.error("Failed to import ultralytics. Please install with: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def detect(self, image: np.ndarray) -> List[List[float]]:
        """
        Detect objects in image
        
        Args:
            image: Input image in BGR format (HWC)
            
        Returns:
            List of detections [x1, y1, x2, y2, conf, class_id]
        """
        if image is None or not isinstance(image, np.ndarray):
            logger.error("Invalid input image")
            return []
            
        try:
            # Make a copy of the image to avoid modifications
            img = image.copy()
            
            # Run inference
            start_time = time.time()
            results = self.model(img, verbose=False)
            inference_time = (time.time() - start_time) * 1000
            logger.debug(f"Inference time: {inference_time:.2f}ms")
            
            # Convert results to standard format [x1, y1, x2, y2, conf, class_id]
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                if len(boxes) == 0:
                    continue
                    
                # Get boxes, confidence scores and class IDs
                xyxy = boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy()
                
                # Combine into detections list
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    conf = confs[i]
                    cls_id = cls_ids[i]
                    detections.append([x1, y1, x2, y2, conf, cls_id])
            
            return detections
            
        except Exception as e:
            logger.exception(f"Error during detection: {e}")
            return []


    def draw_detections(self, image: np.ndarray, detections: List[List[float]]) -> np.ndarray:
        """
        Draw detection bounding boxes on image
        
        Args:
            image: Input image
            detections: List of detections [x1, y1, x2, y2, conf, class_id]
            
        Returns:
            Image with drawn detections
        """
        img = image.copy()
        
        # 缺陷标记使用红色
        defect_color = (0, 0, 255)  # 红色用于标记缺陷区域
        
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            
            # Convert to int
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 计算缺陷区域（仅标记估计的缺陷区域而不是整个物体）
            box_width = x2 - x1
            box_height = y2 - y1
            
            # 假设缺陷通常位于检测框的中心区域，仅标记中心部分
            # 可以根据实际情况调整这个比例
            center_ratio = 0.6  # 中心区域占整个框的比例
            
            defect_x1 = int(x1 + (box_width * (1 - center_ratio) / 2))
            defect_y1 = int(y1 + (box_height * (1 - center_ratio) / 2))
            defect_x2 = int(x2 - (box_width * (1 - center_ratio) / 2))
            defect_y2 = int(y2 - (box_height * (1 - center_ratio) / 2))
            
            # 标记精确的缺陷区域
            cv2.rectangle(img, (defect_x1, defect_y1), (defect_x2, defect_y2), defect_color, 2)
            
            # 绘制缺陷标签
            label = f"缺陷 {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (defect_x1, defect_y1 - label_h - 5), (defect_x1 + label_w, defect_y1), defect_color, -1)
            cv2.putText(img, label, (defect_x1, defect_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img

    def draw_detection(self, image: np.ndarray, detection: List[float], color=(0, 0, 255)) -> np.ndarray:
        """
        在图像上绘制单个检测框
        
        Args:
            image: 输入图像
            detection: [x1, y1, x2, y2, conf, class_id]格式的检测结果
            color: BGR颜色元组
        
        Returns:
            带有检测框的图像
        """
        if image is None or len(detection) < 6:
            return image
            
        img = image.copy()
        x1, y1, x2, y2 = map(int, detection[:4])
        conf = float(detection[4])
        class_id = int(detection[5])
        
        # 绘制矩形框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 添加类别标签和置信度
        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"类别{class_id}"
        label = f"{class_name}: {conf:.2f}"
        
        # 获取标签文本大小
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # 绘制标签背景
        cv2.rectangle(img, (x1, y1 - text_height - 5), (x1 + text_width + 5, y1), color, -1)
        
        # 绘制标签文本
        cv2.putText(img, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return img
# For testing
# For testing
# if __name__ == "__main__":
#     # Configure logging
#     logging.basicConfig(
#         level=logging.DEBUG,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )
    
#     try:
#         # Initialize detector with YOLOv11n model
#         # Options include: yolov11n, yolov11s, yolov11m, yolov11l
#         detector = YOLODetector(
#             model_name="yolo11l", 
#             conf_thresh=0.25
#         )
        
#         # Test with local image if available
#         test_img_path = "../data/images/bus.jpg"
#         if not os.path.exists(test_img_path):
#             logger.warning(f"Test image not found: {test_img_path}")
#             # Try to find another image
#             for img_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
#                 found_imgs = [f for f in os.listdir('.') if f.lower().endswith(img_ext)]
#                 if found_imgs:
#                     test_img_path = found_imgs[0]
#                     logger.info(f"Using alternative test image: {test_img_path}")
#                     break
#             else:
#                 logger.error("No test images found")
#                 exit(1)
        
#         # Load and process image
#         img = cv2.imread(test_img_path)
#         logger.info(f"Loaded image with shape: {img.shape}")
        
#         # Perform detection
#         start_time = time.time()
#         detections = detector.detect(img)
#         elapsed = (time.time() - start_time) * 1000
        
#         # Log results
#         logger.info(f"Detection completed in {elapsed:.2f}ms")
#         logger.info(f"Found {len(detections)} objects")
        
#         # Draw detections
#         result_img = detector.draw_detections(img, detections)
        
#         # Save and display result
#         output_path = "detection_result.jpg"
#         cv2.imwrite(output_path, result_img)
#         logger.info(f"Result saved to {output_path}")
        
#     except Exception as e:
#         logger.exception(f"Error during test: {e}")