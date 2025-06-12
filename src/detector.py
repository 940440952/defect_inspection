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
        在图像上绘制多个检测框
        
        Args:
            image: 输入图像
            detections: 检测结果列表 [x1, y1, x2, y2, conf, class_id]
            
        Returns:
            带有检测框的图像
        """
        img = image.copy()
        for det in detections:
            img = self.draw_detection(img, det)
        return img

    def draw_detection(self, image: np.ndarray, detection: List[float]) -> np.ndarray:
        """
        在图像上绘制单个检测框，支持中文标签
        
        Args:
            image: 输入图像
            detection: [x1, y1, x2, y2, conf, class_id]格式的检测结果
            
        Returns:
            带有检测框的图像
        """
        if image is None or len(detection) < 6:
            return image
            
        img = image.copy()
        x1, y1, x2, y2 = map(int, detection[:4])
        conf = float(detection[4])
        class_id = int(detection[5])
        
        # 为不同类型定义颜色 (BGR格式)
        colors = [
            (0, 0, 255),     # 红色 - 小划痕
            (0, 127, 255),   # 橙色 - 小污点
            (0, 255, 0),     # 绿色 - 大划痕
            (255, 0, 0),     # 蓝色 - 大污点
            (255, 0, 255),   # 紫色 - 堵孔
            (255, 255, 0),   # 青色 - 其他类型
        ]
        
        # 选择颜色 (防止索引越界)
        color = colors[class_id % len(colors)]
        
        # 绘制矩形框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 使用PIL绘制中文文本
        try:
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
            
            # 获取类别名称
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"类别{class_id}"
            label = f"{class_name}: {conf:.2f}"
            
            # 转换为PIL图像
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            # 尝试加载中文字体
            font_paths = [
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Ubuntu
                "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",             # 文泉驿微米黑
                "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",                 # 文泉驿正黑
                "/usr/share/fonts/noto/NotoSansCJK-Regular.ttc",              # Noto Sans CJK
                "C:/Windows/Fonts/simhei.ttf",                                # Windows 黑体
                "C:/Windows/Fonts/simfang.ttf",                               # Windows 仿宋
            ]
            
            # 尝试加载字体，如果都失败则使用默认
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, 20)
                        break
                    except Exception:
                        continue
                        
            if font is None:
                font = ImageFont.load_default()
                
            # 测量文本尺寸
            try:
                # 在新版PIL中使用textbbox
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except AttributeError:
                # 在旧版PIL中使用textsize
                text_width, text_height = draw.textsize(label, font=font)
            
            # 绘制文本背景
            if y1 - text_height - 5 > 0:
                # 文本放在框上方
                rect_y1 = y1 - text_height - 5
                rect_y2 = y1
                text_y = rect_y1
            else:
                # 文本放在框内上方
                rect_y1 = y1
                rect_y2 = y1 + text_height + 5
                text_y = rect_y1
                
            # 绘制背景矩形（使用OpenCV，因为PIL的矩形绘制没有透明度选项）
            cv2.rectangle(img, (x1, rect_y1), (x1 + text_width + 10, rect_y2), color, -1)
            
            # 将更新后的图像转换回PIL
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            # 绘制文本
            draw.text((x1 + 5, text_y + 1), label, font=font, fill=(255, 255, 255))
            
            # 转换回OpenCV图像
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
        except ImportError:
            # 如果没有PIL，回退到OpenCV的英文绘制
            logger.warning("未找到PIL库，中文标签可能无法正常显示")
            class_name = f"ID:{class_id}"  # 使用ID代替中文名
            label = f"{class_name}: {conf:.2f}"
            
            # 获取文本尺寸
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # 绘制背景矩形
            cv2.rectangle(
                img, (x1, y1 - text_height - 5), 
                (x1 + text_width + 5, y1), color, -1
            )
            
            # 绘制文本
            cv2.putText(
                img, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
            
        return img

# For testing
# if __name__ == "__main__":
#     import cv2
#     import logging
#     import os
#     from pathlib import Path
    
#     # 配置日志
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )
    
#     # 测试参数 - 硬编码，无需命令行参数
#     test_image_path = "/home/gtm/defect_inspection/image/00a9ab84-image_1746153465.jpg"  # 测试图像路径
#     models_dir = "/home/gtm/defect_inspection/models/detector"  # 模型目录
#     model_name = "detector"  # 模型名称
#     min_area = 100  # 最小面积阈值
#     confidence_threshold = 0.25  # 置信度阈值
#     class_names = ["小划痕", "小污点", "大划痕", "大污点", "堵孔"]  # 类别名称
#     output_dir = "/home/gtm/defect_inspection/test_results"  # 输出目录
    
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 检查测试图像是否存在，不存在则创建一个简单的测试图像
#     if not os.path.exists(test_image_path):
#         print(f"测试图像 {test_image_path} 不存在，创建一个简单的测试图像")
#         test_img = np.ones((640, 640, 3), dtype=np.uint8) * 255  # 白色背景
#         cv2.rectangle(test_img, (100, 100), (300, 300), (0, 0, 255), 2)  # 红色矩形
#         cv2.rectangle(test_img, (350, 350), (400, 400), (255, 0, 0), 2)  # 蓝色矩形
#         cv2.imwrite(test_image_path, test_img)
    
#     try:
#         print(f"开始测试 YOLODetector...")
#         print(f"测试参数: 模型名称={model_name}, 模型目录={models_dir}")
#         print(f"过滤参数: 最小面积={min_area}, 置信度阈值={confidence_threshold}")
        
#         # 初始化检测器
#         detector = YOLODetector(
#             model_name=model_name,
#             models_dir=models_dir,
#             filter_enabled=True,
#             min_area=min_area,
#             confidence_threshold=confidence_threshold
#         )
        
#         # 设置类别名称
#         detector.class_names = class_names
#         print(f"设置类别名称: {class_names}")
        
#         # 加载测试图像
#         print(f"加载测试图像: {test_image_path}")
#         image = cv2.imread(test_image_path)
#         if image is None:
#             print(f"无法加载测试图像: {test_image_path}")
#             exit(1)
        
#         # 转换为RGB (YOLO使用RGB格式)
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         # 打印图像信息
#         height, width, channels = image.shape
#         print(f"图像尺寸: {width} x {height} x {channels}")
        
#         print("----------- 测试未过滤检测 -----------")
#         # 执行检测 - 不过滤
#         detections_no_filter = detector.detect(image_rgb, apply_filter=False)
#         print(f"检测到 {len(detections_no_filter)} 个结果 (未过滤)")
        
#         # 打印未过滤检测结果
#         for i, det in enumerate(detections_no_filter):
#             x1, y1, x2, y2, conf, cls_id = det
#             cls_id_int = int(cls_id)
#             class_name = detector.class_names[cls_id_int] if cls_id_int < len(detector.class_names) else f"类别{cls_id_int}"
#             area = (x2 - x1) * (y2 - y1)
#             print(f"[{i+1}] 类别: {class_name}, 置信度: {conf:.2f}, 面积: {area:.1f}像素, 坐标: {x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}")
        
#         print("\n----------- 测试过滤检测 -----------")
#         # 执行检测 - 应用过滤
#         detections_filtered = detector.detect(image_rgb, apply_filter=True)
#         print(f"检测到 {len(detections_filtered)} 个结果 (已过滤)")
        
#         # 打印已过滤检测结果
#         for i, det in enumerate(detections_filtered):
#             x1, y1, x2, y2, conf, cls_id = det
#             cls_id_int = int(cls_id)
#             class_name = detector.class_names[cls_id_int] if cls_id_int < len(detector.class_names) else f"类别{cls_id_int}"
#             area = (x2 - x1) * (y2 - y1)
#             print(f"[{i+1}] 类别: {class_name}, 置信度: {conf:.2f}, 面积: {area:.1f}像素, 坐标: {x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}")
        
#         print("\n----------- 生成测试结果图像 -----------")
#         # 绘制未过滤的检测结果
#         result_no_filter = image.copy()
#         for det in detections_no_filter:
#             result_no_filter = detector.draw_detection(result_no_filter, det)
        
#         # 绘制已过滤的检测结果
#         result_filtered = image.copy()
#         for det in detections_filtered:
#             result_filtered = detector.draw_detection(result_filtered, det)
        
#         # 保存结果图像
#         no_filter_path = os.path.join(output_dir, "result_no_filter.jpg")
#         filtered_path = os.path.join(output_dir, "result_filtered.jpg")
#         cv2.imwrite(no_filter_path, result_no_filter)
#         cv2.imwrite(filtered_path, result_filtered)
        
#         print(f"未过滤结果图像已保存: {no_filter_path}")
#         print(f"已过滤结果图像已保存: {filtered_path}")
        
#         # 打印测试结果摘要
#         print("\n----------- 测试结果摘要 -----------")
#         print(f"总检测结果数: {len(detections_no_filter)}")
#         print(f"过滤后剩余结果数: {len(detections_filtered)}")
#         print(f"过滤掉的结果数: {len(detections_no_filter) - len(detections_filtered)}")
        
#         print("\n测试成功完成！")
        
#     except Exception as e:
#         import traceback
#         print(f"测试过程中发生错误:")
#         traceback.print_exc()
