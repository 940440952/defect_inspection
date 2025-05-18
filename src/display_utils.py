#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
显示工具函数模块
处理图像绘制、裁剪区域和检测结果的可视化
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any


def draw_combined_detections(
    image: np.ndarray, 
    crop_results: List[Dict], 
    detection_results_dict: Dict[int, List],
    class_names: List[str] = None,
    crop_color: Tuple[int, int, int] = (0, 255, 0),  # 裁剪框颜色(绿色)
    defect_color: Tuple[int, int, int] = (0, 0, 255),  # 瑕疵框颜色(红色)
    thickness: int = 2,
    font_scale: float = 1,
    show_confidence: bool = True
) -> np.ndarray:
    """
    在原始图像上绘制裁剪区域和检测结果
    
    Args:
        image: 原始输入图像
        crop_results: 裁剪结果列表，每项包含'box'键为裁剪坐标[x1,y1,x2,y2]
        detection_results_dict: 检测结果字典，键为裁剪区域索引，值为该区域内的检测结果列表
        class_names: 类别名称列表，默认为None
        crop_color: 裁剪框颜色，默认绿色
        defect_color: 瑕疵框颜色，默认红色
        thickness: 线条粗细
        font_scale: 字体大小
        show_confidence: 是否显示置信度
        
    Returns:
        带有绘制结果的图像
    """
    if image is None:
        return None
        
    # 复制图像以避免修改原始图像
    result_image = image.copy()
    
    # 如果没有提供类别名称，使用默认值
    if class_names is None:
        class_names = ["defect"]
    
    # 1. 首先绘制所有裁剪区域框
    for i, crop_result in enumerate(crop_results):
        crop_box = [int(coord) for coord in crop_result['box']]
        x1, y1, x2, y2 = crop_box
        
        # 绘制裁剪框
        cv2.rectangle(
            result_image, 
            (x1, y1), (x2, y2), 
            crop_color, 
            thickness
        )
        
        # 添加裁剪区域编号
        label = f"区域 #{i+1}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # 绘制标签背景
        cv2.rectangle(
            result_image, 
            (x1, y1 - text_height - 5), 
            (x1 + text_width + 5, y1), 
            crop_color, 
            -1  # 填充矩形
        )
        
        # 绘制标签文本
        cv2.putText(
            result_image, 
            label, 
            (x1 + 3, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            (0, 0, 0), 
            thickness
        )
        
        # 2. 绘制该裁剪区域内的瑕疵检测结果
        detections = detection_results_dict.get(i, [])
        for det in detections:
            # 检测框原始坐标在裁剪图内，需要转换到原图坐标系
            det_x1 = int(det[0]) + x1
            det_y1 = int(det[1]) + y1
            det_x2 = int(det[2]) + x1
            det_y2 = int(det[3]) + y1
            confidence = float(det[4])
            class_id = int(det[5]) if len(det) > 5 else 0
            
            # 绘制瑕疵框
            cv2.rectangle(
                result_image, 
                (det_x1, det_y1), (det_x2, det_y2), 
                defect_color, 
                thickness
            )
            
            # 准备标签文本
            class_name = class_names[class_id] if class_id < len(class_names) else f"类别{class_id}"
            if show_confidence:
                label = f"{class_name}: {confidence:.2f}"
            else:
                label = class_name
                
            # 获取文本大小
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # 绘制标签背景
            cv2.rectangle(
                result_image, 
                (det_x1, det_y1 - text_height - 5), 
                (det_x1 + text_width + 5, det_y1), 
                defect_color, 
                -1  # 填充矩形
            )
            
            # 绘制标签文本
            cv2.putText(
                result_image, 
                label, 
                (det_x1 + 3, det_y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (255, 255, 255),  # 白色文本
                thickness
            )
    
    # 3. 如果有未与裁剪区域关联的检测结果，直接绘制
    if -1 in detection_results_dict:
        for det in detection_results_dict[-1]:
            x1, y1, x2, y2 = [int(coord) for coord in det[:4]]
            confidence = float(det[4])
            class_id = int(det[5]) if len(det) > 5 else 0
            
            # 绘制检测框
            cv2.rectangle(
                result_image, 
                (x1, y1), (x2, y2), 
                defect_color, 
                thickness
            )
            
            # 准备标签文本
            class_name = class_names[class_id] if class_id < len(class_names) else f"类别{class_id}"
            if show_confidence:
                label = f"{class_name}: {confidence:.2f}"
            else:
                label = class_name
                
            # 获取文本大小
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # 绘制标签背景
            cv2.rectangle(
                result_image, 
                (x1, y1 - text_height - 5), 
                (x1 + text_width + 5, y1), 
                defect_color, 
                -1  # 填充矩形
            )
            
            # 绘制标签文本
            cv2.putText(
                result_image, 
                label, 
                (x1 + 3, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (255, 255, 255),  # 白色文本
                thickness
            )
    
    return result_image


def merge_detection_results(
    crop_results: List[Dict], 
    crop_detections: List[List[List[float]]]
) -> Dict[int, List]:
    """
    将每个裁剪区域的检测结果组织成字典形式
    
    Args:
        crop_results: 裁剪结果列表
        crop_detections: 每个裁剪区域的检测结果列表
    
    Returns:
        字典，键为裁剪区域索引，值为该区域内的检测结果列表
    """
    detection_dict = {}
    
    for i, detections in enumerate(crop_detections):
        if detections:
            detection_dict[i] = detections
    
    return detection_dict