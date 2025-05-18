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
             models_dir: str = "../models",
             filter_enabled: bool = True,
             min_area: float = 100,
             confidence_threshold: float = 0.25):
        """
        Initialize YOLO detector
        
        Args:
            model_name: Base name of the model (e.g. "yolov11n", "yolov11s", "yolov11m", "yolov11l")
            conf_thresh: Confidence threshold for detections
            nms_thresh: NMS IoU threshold
            models_dir: Directory to store and load models
        """
        self.model_name = model_name
        self.models_dir = models_dir
        self.model = None
        self.filter_enabled = filter_enabled
        self.min_area = min_area
        self.confidence_threshold = confidence_threshold
        self.class_names = ["deformation"]
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load model
        self._load_model()
        logger.info(f"YOLO detector initialized with model: {self.model_name}")

        # 记录过滤设置
        if self.filter_enabled:
            logger.info(f"检测结果过滤已启用: 最小面积={self.min_area}像素, 最小置信度={self.confidence_threshold:.2f}")
        else:
            logger.info("检测结果过滤已禁用")
            
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
    
    def detect(self, image: np.ndarray, apply_filter: bool = None) -> List[List[float]]:
        """
        在图像中检测物体，并根据初始化参数自动进行过滤
        
        Args:
            image: 输入图像 (BGR格式, HWC)
            apply_filter: 是否应用过滤，None表示使用初始化时的filter_enabled设置
            
        Returns:
            检测结果列表，格式为 [[x1, y1, x2, y2, conf, class_id], ...]
        """
        if image is None or not isinstance(image, np.ndarray):
            logger.error("无效的输入图像")
            return []
            
        try:
            # 创建图像副本避免修改原图
            img = image.copy()
            
            # 运行推理
            start_time = time.time()
            results = self.model(img, verbose=False)
            inference_time = (time.time() - start_time) * 1000
            logger.debug(f"推理时间: {inference_time:.2f}ms")
            
            # 转换结果为标准格式 [x1, y1, x2, y2, conf, class_id]
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                if len(boxes) == 0:
                    continue
                    
                # 获取检测框、置信度和类别ID
                xyxy = boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy()
                
                # 组合为检测结果列表
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    conf = confs[i]
                    cls_id = cls_ids[i]
                    detections.append([x1, y1, x2, y2, conf, cls_id])
            
            # 确定是否应用过滤
            should_filter = self.filter_enabled if apply_filter is None else apply_filter
            
            # 如果需要过滤，就应用过滤条件
            if should_filter and detections:
                # 记录原始检测数量
                orig_count = len(detections)
                filtered_results = []
                filtered_count = 0
                
                for detection in detections:
                    x1, y1, x2, y2, conf, class_id = detection
                    
                    # 计算面积
                    area = (x2 - x1) * (y2 - y1)
                    
                    # 应用过滤条件
                    area_ok = area >= self.min_area
                    conf_ok = conf >= self.confidence_threshold
                    
                    if area_ok and conf_ok:
                        filtered_results.append(detection)
                    else:
                        filtered_count += 1
                        class_id_int = int(class_id)
                        class_name = self.class_names[class_id_int] if class_id_int < len(self.class_names) else f"类别{class_id_int}"
                        logger.debug(f"过滤掉检测结果: 类别={class_name}, "
                                    f"面积={area:.1f}像素, 置信度={conf:.2f}")
                
                if filtered_count > 0:
                    logger.info(f"过滤掉 {filtered_count}/{orig_count} 个检测结果 "
                            f"(面积<{self.min_area} 或 置信度<{self.confidence_threshold:.2f})")
                    
                detections = filtered_results
            
            return detections
            
        except Exception as e:
            logger.exception(f"检测过程出错: {e}")
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