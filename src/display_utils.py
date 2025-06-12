#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
显示工具函数模块
处理图像绘制、裁剪区域和检测结果的可视化
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any


def draw_text_with_pil(image, text, position, font_size=20, color=(255, 255, 255), bg_color=None, stroke_width=1):
    """
    使用PIL在图像上绘制中文文本，解决中文和数字混合显示问题
    
    Args:
        image: 输入图像(OpenCV格式,BGR)
        text: 要绘制的文本
        position: 文本位置(x, y)
        font_size: 字体大小
        color: 文本颜色(RGB)
        bg_color: 背景颜色(RGB)，None表示无背景
        stroke_width: 文字描边宽度，0表示无描边
        
    Returns:
        绘制了文本的图像(OpenCV格式,BGR)
    """
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    import os
    
    # 转换为PIL图像
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    # 确保文本是字符串类型
    text = str(text)
    
    # 先尝试加载支持中文的字体
    chinese_font_paths = [
        "/usr/share/fonts/noto/NotoSansCJK-Regular.ttc",              # Ubuntu Noto
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",     # 另一种 Noto 路径
        "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.otf",      # 思源黑体
        "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",             # 文泉驿微米黑
        "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",                 # 文泉驿正黑
        "C:/Windows/Fonts/simhei.ttf",                                # Windows 黑体
        "C:/Windows/Fonts/arial.ttf",                                 # Arial
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",            # DejaVu

    ]
    
    # 尝试加载中文字体
    font = None
    for font_path in chinese_font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except Exception:
                continue
    
    # 如果中文字体加载失败，尝试加载任何系统字体
    if font is None:
        try:
            # 在Linux上尝试使用fc-list查找字体
            import subprocess
            fonts_list = subprocess.check_output(['fc-list', ':lang=zh', 'file']).decode('utf-8').strip().split('\n')
            if fonts_list:
                for font_line in fonts_list:
                    font_path = font_line.split(':')[0]
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except:
                        continue
        except:
            pass
    
    # 如果还是无法加载字体，使用默认字体
    if font is None:
        font = ImageFont.load_default()
        # 放大默认字体，以便更清晰
        font_size = max(10, font_size // 2)
    
    # 测量文本尺寸
    try:
        # 新版PIL使用textbbox
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        # 旧版PIL使用textsize
        text_width, text_height = draw.textsize(text, font=font)
    
    # 绘制背景（如果指定了背景颜色）
    x, y = position
    if bg_color is not None:
        bg_x1, bg_y1 = x, y
        bg_x2, bg_y2 = x + text_width, y + text_height
        draw.rectangle([(bg_x1, bg_y1), (bg_x2, bg_y2)], fill=bg_color)
    
    # 绘制文本（带描边）
    if stroke_width > 0:
        draw.text((x, y), text, font=font, fill=color, stroke_width=stroke_width, stroke_fill=(0, 0, 0))
    else:
        draw.text((x, y), text, font=font, fill=color)
    
    # 转换回OpenCV图像
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def draw_combined_detections(
    image: np.ndarray, 
    crop_results: List[Dict], 
    detection_results_dict: Dict[int, List],
    class_names: List[str] = None,
    crop_color: Tuple[int, int, int] = (255, 255, 255),  # 裁剪框颜色(绿色)
    thickness: int = 2,
    font_scale: float = 1,
    show_confidence: bool = True,
    show_labels: bool = True
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
        
        # 只有启用标签显示时才绘制
        if show_labels:
            # 添加裁剪区域编号
            label = f"区域 {i+1}"
            # 使用PIL绘制中文标签
            result_image = draw_text_with_pil(
                result_image,
                label,
                (x1 + 3, y1 - int(font_scale * 30)),
                font_size=int(font_scale * 20),
                color=(255, 255, 255),  # 白色文本
                bg_color=None,
                stroke_width=2  # 添加描边，提高可读性
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
            det_color = colors[class_id % len(colors)]
            
            # 绘制瑕疵框
            cv2.rectangle(
                result_image, 
                (det_x1, det_y1), (det_x2, det_y2), 
                det_color, 
                thickness
            )
            
            # 只有启用标签显示时才绘制
            if show_labels:
                # 准备标签文本
                class_name = class_names[class_id] if class_id < len(class_names) else f"类别{class_id}"
                if show_confidence:
                    confidence_value = float(confidence)
                    confidence_str = f"{confidence_value:.2f}"
                    label = f"{class_name}:{confidence_str}"
                else:
                    label = class_name
                    
                # 使用PIL绘制中文标签
                result_image = draw_text_with_pil(
                    result_image,
                    label,
                    (det_x1 + 3, det_y1 - int(font_scale * 30)),
                    font_size=int(font_scale * 20),
                    color=(255, 255, 255),  # 白色文本
                    bg_color=None,
                    stroke_width=1  # 添加描边，提高可读性
                )
    
    # 3. 如果有未与裁剪区域关联的检测结果，直接绘制
    if -1 in detection_results_dict:
        for det in detection_results_dict[-1]:
            x1, y1, x2, y2 = [int(coord) for coord in det[:4]]
            confidence = float(det[4])
            class_id = int(det[5]) if len(det) > 5 else 0
            
            # 选择颜色
            det_color = colors[class_id % len(colors)]
            
            # 绘制检测框
            cv2.rectangle(
                result_image, 
                (x1, y1), (x2, y2), 
                det_color, 
                thickness
            )
            
            # 只有启用标签显示时才绘制
            if show_labels:
                # 准备标签文本
                class_name = class_names[class_id] if class_id < len(class_names) else f"类别{class_id}"
                if show_confidence:
                    confidence_value = float(confidence)
                    confidence_str = f"{confidence_value:.2f}"
                    label = f"{class_name}:{confidence_str}"
                else:
                    label = class_name
                    
                # 使用PIL绘制中文标签
                result_image = draw_text_with_pil(
                    result_image,
                    label,
                    (det_x1 + 3, det_y1 -  int(font_scale * 30)),
                    font_size=int(font_scale * 20),
                    color=(255, 255, 255),  # 白色文本
                    bg_color=None,
                    stroke_width=1  # 添加描边，提高可读性
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