#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像裁剪工具模块

用于根据YOLO检测结果裁剪高分辨率图像中的特定区域，主要功能：
1. 加载YOLO模型进行缺陷检测
2. 根据检测框裁剪图像
"""

import cv2
import numpy as np
import logging
import os
import shutil
import time
from typing import List, Dict, Optional, Tuple, Union

# 设置日志
logger = logging.getLogger("DefectInspection.ImageCropper")

class ImageCropper:
    """
    图像裁剪工具类
    
    用于使用YOLO检测并裁剪图像中的缺陷区域
    """
    
    def __init__(self, 
                 model_name: str = "yolo11n", 
                 conf_threshold: float = 0.25,
                 nms_threshold: float = 0.45,
                 models_dir: str = "../models",
                 use_dla: bool = False,
                 dla_core: int = 0):
        """
        初始化图像裁剪工具
        
        Args:
            model_name: 模型名称（如"yolov11n", "yolov11s", "yolov5s"等）
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
            models_dir: 模型目录
            use_dla: 是否使用DLA加速（仅Jetson设备支持）
            dla_core: DLA核心编号（0或1）
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.models_dir = models_dir
        self.use_dla = use_dla
        self.dla_core = dla_core
        self.model = None
        self.class_names = ["defect"]
        
        # 创建模型目录（如果不存在）
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 加载模型
        self._load_model()
        logger.info(f"图像裁剪工具初始化完成，使用模型：{self.model_name}")
    
    def _load_model(self):
        """加载模型，直接使用PyTorch模型，不转换为TensorRT"""
        try:
            # 导入YOLO库
            from ultralytics import YOLO
            
            # 检查PyTorch模型路径
            pt_path = os.path.join(self.models_dir, f"{self.model_name}.pt")
            
            print(f"cropper模型路径: {pt_path}")
            
            # 情况1：PyTorch模型存在，直接加载
            if os.path.exists(pt_path):
                logger.info(f"加载PyTorch模型: {pt_path}")
                self.model = YOLO(pt_path)
                
            # 情况2：从Ultralytics下载
            else:
                logger.info(f"未发现本地模型，从Ultralytics下载 {self.model_name}")
                
                # 下载模型
                self.model = YOLO(self.model_name)
                
                # 保存PyTorch模型以供将来使用
                self.model.save(pt_path)
                logger.info(f"已保存PyTorch模型到 {pt_path}")
            
            # 设置模型参数
            self.model.conf = self.conf_threshold  # 检测置信度阈值
            self.model.iou = self.nms_threshold    # NMS IoU阈值
            
            # 运行预热推理
            logger.info("进行预热推理")
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_input)
            logger.info("PyTorch模型已加载并准备好进行推理")
            
        except ImportError:
            logger.error("导入ultralytics失败。请使用命令安装：pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"加载模型时出错：{e}")
            raise
    
    def detect(self, image: np.ndarray) -> List[List[float]]:
        """
        使用YOLO模型检测图像中的缺陷
        
        Args:
            image: 输入图像
            
        Returns:
            检测结果列表 [[x1, y1, x2, y2, conf, class_id], ...]
        """
        if image is None or not isinstance(image, np.ndarray):
            logger.error("无效的输入图像")
            return []
            
        try:
            # 复制图像以避免修改
            img = image.copy()
            
            # 运行推理
            start_time = time.time()
            results = self.model(img, verbose=False)
            inference_time = (time.time() - start_time) * 1000
            logger.debug(f"推理时间：{inference_time:.2f}ms")
            
            # 将结果转换为标准格式 [x1, y1, x2, y2, conf, class_id]
            detections = []
            for result in results:
                boxes = result.boxes
                
                if len(boxes) == 0:
                    continue
                    
                # 获取边界框、置信度分数和类别ID
                xyxy = boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy()
                
                # 组合为检测结果列表
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    conf = confs[i]
                    cls_id = cls_ids[i]
                    detections.append([x1, y1, x2, y2, conf, cls_id])
            
            return detections
            
        except Exception as e:
            logger.exception(f"检测过程中出错：{e}")
            return []
        
    def crop_by_detection(self, image: np.ndarray, 
                         detection: List[float]) -> np.ndarray:
        """
        根据单个检测结果裁剪图像
        
        Args:
            image: 原始图像 (OpenCV格式，BGR)
            detection: YOLO检测结果 [x1, y1, x2, y2, conf, class_id]
            
        Returns:
            裁剪后的图像
        """
        if image is None or len(detection) < 4:
            logger.error("无效的图像或检测结果")
            return None
            
        # 获取图像尺寸
        img_height, img_width = image.shape[:2]
        
        # 获取检测框坐标
        x1, y1, x2, y2 = [int(coord) for coord in detection[:4]]
        
        # 确保坐标在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)
        
        # 裁剪图像
        try:
            if x1 >= x2 or y1 >= y2:
                logger.error("无效的裁剪坐标")
                return None
                
            cropped_img = image[y1:y2, x1:x2]
            return cropped_img
        except Exception as e:
            logger.error(f"裁剪图像失败: {str(e)}")
            return None
    
    def crop_detection(self, image: np.ndarray, 
                      detection: List[float]) -> Optional[Dict]:
        """
        裁剪单个检测结果
        
        Args:
            image: 原始图像
            detection: YOLO检测结果 [x1, y1, x2, y2, conf, class_id]
            
        Returns:
            裁剪结果字典 {'crop': 裁剪图像, 'box': [x1,y1,x2,y2], 'conf': 置信度, 'class_id': 类别ID}
            如果裁剪失败则返回None
        """
        if image is None or len(detection) < 6:
            return None
            
        # 裁剪图像
        cropped = self.crop_by_detection(image, detection)
        
        if cropped is not None:
            # 返回裁剪结果
            result = {
                'crop': cropped,
                'box': detection[:4],
                'conf': float(detection[4]),
                'class_id': int(detection[5])
            }
            return result
        
        return None
    
    def detect_and_crop(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的缺陷并裁剪
        
        Args:
            image: 输入图像
            
        Returns:
            裁剪结果列表，每项为 {'crop': 裁剪图像, 'box': [x1,y1,x2,y2], 'conf': 置信度, 'class_id': 类别ID}
        """
        if image is None:
            logger.error("输入图像为空")
            return []
            
        # 检测缺陷
        detections = self.detect(image)
        
        if not detections:
            logger.info("未检测到缺陷")
            return []
            
        # 裁剪每个检测结果
        crop_results = []
        for detection in detections:
            result = self.crop_detection(image, detection)
            if result:
                crop_results.append(result)
                
        logger.info(f"已检测并裁剪{len(crop_results)}个缺陷区域")
        return crop_results
    
    def process_image_file(self, image_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        处理图像文件：加载、检测并裁剪
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            (原始图像, 裁剪结果列表)
        """
        try:
            # 加载图像
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法加载图像: {image_path}")
                return None, []
                
            # 检测并裁剪
            crop_results = self.detect_and_crop(image)
            return image, crop_results
            
        except Exception as e:
            logger.error(f"处理图像失败: {str(e)}")
            return None, []


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 初始化裁剪工具，使用YOLOv11模型
    cropper = ImageCropper(
        model_name="best", 
        conf_threshold=0.25,
        models_dir="models/cropper",
        use_dla=False  # 在Jetson平台上可设为True
    )
    
    # 处理测试图像
    test_img_path = "/home/gtm/defect_inspection/image/0a3a8c16-image_1745736852.jpg"
    if os.path.exists(test_img_path):
        original_image, cropped_results = cropper.process_image_file(test_img_path)
        
        # 显示结果
        if original_image is not None:
            print(f"处理图像: {test_img_path}, 尺寸: {original_image.shape}")
            print(f"检测到 {len(cropped_results)} 个缺陷区域")
            
            # 保存裁剪结果
            for i, result in enumerate(cropped_results):
                output_path = f"cropped_{i}.jpg"
                cv2.imwrite(output_path, result['crop'])
                print(f"裁剪结果 {i}: 位置 {result['box']}, 置信度: {result['conf']:.2f}, 类别: {result['class_id']}")
    else:
        print(f"测试图像 {test_img_path} 不存在")