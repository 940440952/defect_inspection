#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
瑕疵检测系统主程序

系统流程:
1. 位置传感器检测到产品到达检测区域
2. 控制传送带停止运行
3. 摄像头捕获产品图像
4. YOLO模型进行瑕疵检测
5. 照片和检测结果上传服务器
6. 根据检测结果决定是否激活剔除机构
7. 重启传送带，继续下一个产品的检测
"""

import os
import sys
import time
import logging
import argparse
import threading
import cv2
import numpy as np
from datetime import datetime

# 导入项目模块
from src.utils import setup_logging, load_config, get_timestamp, ensure_directory_exists
from src.io_controller import IOController, IOMode
from src.detector import YOLODetector
from src.api_client import APIClient
from src.display import DisplayInterface

# 全局变量
config = None
io_controller = None
camera = None
detector = None
api_client = None
display = None
logger = None
running = True
inspection_lock = threading.Lock()

def initialize_system(config_path):
    """初始化系统组件"""
    global config, io_controller, camera, detector, api_client, display, logger
    
    # 加载配置
    config = load_config(config_path)
    
    # 设置日志
    log_dir = config.get("log_dir", "logs")
    ensure_directory_exists(log_dir)
    log_file = os.path.join(log_dir, f"defect_inspection_{get_timestamp()}.log")
    log_level = getattr(logging, config.get("log_level", "INFO").upper())
    logger = setup_logging(level=log_level, log_file=log_file)
    logger.info("系统初始化中...")
    
    # 初始化IO控制器
    io_mode = IOMode.GPIO if config.get("use_gpio", True) else IOMode.SIMULATE
    io_controller = IOController(
        mode=io_mode,
        position_sensor_pin=config.get("pins", {}).get("position_sensor", 18),
        conveyor_control_pin=config.get("pins", {}).get("conveyor_control", 23),
        rejector_control_pin=config.get("pins", {}).get("rejector_control", 24)
    )
    
    # 初始化摄像头
    camera_id = config.get("camera", {}).get("device_id", 0)
    camera = cv2.VideoCapture(camera_id)
    camera_width = config.get("camera", {}).get("width", 1280)
    camera_height = config.get("camera", {}).get("height", 720)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    
    if not camera.isOpened():
        logger.error(f"无法打开摄像头 (ID: {camera_id})")
        sys.exit(1)
    
    # 初始化YOLO检测器
    detector = YOLODetector(
        model_name=config.get("detector", {}).get("model_name", "yolov11n"),
        conf_thresh=config.get("detector", {}).get("conf_threshold", 0.25),
        nms_thresh=config.get("detector", {}).get("nms_threshold", 0.45),
        models_dir=config.get("detector", {}).get("models_dir", "../models"),
        use_dla=config.get("detector", {}).get("use_dla", True)
    )
    
    # 初始化API客户端
    api_client = APIClient(
        api_url=config.get("api", {}).get("url", "http://localhost:8080"),
        api_token=config.get("api", {}).get("token", ""),
        line_name=config.get("api", {}).get("line_name", "LineTest"),
        product_type=config.get("api", {}).get("product_type", "QC")
    )
    
    # 初始化显示界面 (如果需要)
    if not config.get("no_display", False):
        display = DisplayInterface()
        display.start()
    
    # 注册位置传感器回调
    io_controller.register_position_callback(product_detected_callback)
    
    logger.info("系统初始化完成")

def product_detected_callback():
    """位置传感器检测到产品的回调函数"""
    global running
    
    if not running:
        return
    
    # 使用锁防止重复处理
    if not inspection_lock.acquire(blocking=False):
        logger.debug("检测流程正在进行，忽略传感器触发")
        return
    
    try:
        logger.info("检测到产品进入检测区域")
        
        # 步骤1和2：已由传感器检测和回调触发
        
        # 步骤2：停止传送带
        io_controller.stop_conveyor()
        logger.info("传送带已停止")
        
        # 等待传送带完全停止
        time.sleep(config.get("timings", {}).get("conveyor_stop_delay", 0.5))
        
        # 步骤3：捕获图像
        success, image = capture_image()
        if not success:
            logger.error("图像捕获失败，继续下一个产品")
            restart_conveyor()
            return
        
        # 步骤4：执行瑕疵检测
        detection_results = run_detection(image)
        
        # 保存图像和结果
        save_dir = config.get("storage", {}).get("save_dir", "images")
        ensure_directory_exists(save_dir)
        timestamp = get_timestamp()
        image_path = os.path.join(save_dir, f"img_{timestamp}.jpg")
        cv2.imwrite(image_path, image)
        
        # 更新显示界面
        if display:
            display.update_display(image, detection_results)
        
        # 步骤5：上传结果到服务器
        upload_results(image, detection_results, image_path)
        
        # 步骤6：根据检测结果决定是否剔除
        has_defect = process_detection_results(detection_results)
        if has_defect:
            logger.info("检测到瑕疵，激活剔除机构")
            ejection_duration = config.get("ejector", {}).get("duration", 0.5)
            io_controller.activate_rejector(ejection_duration)
        
        # 等待处理完成
        time.sleep(config.get("timings", {}).get("processing_delay", 1.0))
        
        # 步骤7：重启传送带，继续检测下一个产品
        restart_conveyor()
        
    except Exception as e:
        logger.exception(f"产品检测过程中发生错误: {str(e)}")
        # 确保传送带重新启动
        restart_conveyor()
    finally:
        # 释放锁
        inspection_lock.release()

def capture_image():
    """从摄像头捕获图像"""
    # 连续捕获几帧丢弃，确保获取最新图像
    for _ in range(3):
        camera.read()
    
    # 捕获实际使用的图像
    success, frame = camera.read()
    if not success:
        logger.error("无法从摄像头捕获图像")
        return False, None
    
    logger.debug("成功捕获产品图像")
    return True, frame

def run_detection(image):
    """运行YOLO检测"""
    try:
        start_time = time.time()
        detections = detector.detect(image)
        elapsed = time.time() - start_time
        
        logger.info(f"检测完成，耗时 {elapsed:.3f} 秒, 发现 {len(detections)} 个瑕疵")
        return detections
    
    except Exception as e:
        logger.exception(f"检测过程中发生错误: {str(e)}")
        return []

def upload_results(image, detection_results, image_path):
    """上传检测结果到服务器"""
    try:
        # 创建要上传的数据
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "detections": len(detection_results),
            "has_defect": len(detection_results) > 0,
            "image_path": image_path,
        }
        
        # 如果有检测结果，添加更多详细信息
        if detection_results and len(detection_results) > 0:
            defects = []
            for det in detection_results:
                if len(det) >= 6:
                    x1, y1, x2, y2, conf, class_id = det[:6]
                    defects.append({
                        "class_id": int(class_id),
                        "confidence": float(conf),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })
            metadata["defects"] = defects
        
        # 上传到服务器
        success = api_client.upload_inspection_result(image, metadata)
        if success:
            logger.info("成功上传检测结果到服务器")
        else:
            logger.warning("上传检测结果到服务器失败")
            
    except Exception as e:
        logger.exception(f"上传结果时发生错误: {str(e)}")

def process_detection_results(detection_results):
    """处理检测结果，决定是否需要剔除"""
    # 检查是否有任何瑕疵
    if not detection_results or len(detection_results) == 0:
        logger.info("未检测到瑕疵")
        return False
    
    # 获取剔除阈值
    conf_threshold = config.get("ejector", {}).get("confidence_threshold", 0.4)
    
    # 查找置信度最高的瑕疵
    max_conf = 0.0
    for det in detection_results:
        if len(det) >= 5:
            conf = float(det[4])
            max_conf = max(max_conf, conf)
    
    # 决定是否剔除
    should_eject = max_conf >= conf_threshold
    logger.info(f"最高置信度: {max_conf:.3f}, 阈值: {conf_threshold}, 剔除: {should_eject}")
    return should_eject

def restart_conveyor():
    """重新启动传送带"""
    try:
        io_controller.start_conveyor()
        logger.info("传送带已重新启动")
    except Exception as e:
        logger.error(f"重启传送带失败: {str(e)}")

def cleanup():
    """清理资源"""
    global running
    
    running = False
    logger.info("正在清理资源...")
    
    # 停止IO控制器
    if io_controller:
        io_controller.close()
    
    # 释放摄像头
    if camera:
        camera.release()
    
    # 关闭显示界面
    if display:
        display.stop()
    
    logger.info("系统已关闭")

def main():
    """主函数"""
    global running
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="瑕疵检测系统")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--log-level", type=str, default="info", help="日志级别 (debug, info, warning, error)")
    parser.add_argument("--no-display", action="store_true", help="禁用显示界面")
    parser.add_argument("--no-ejection", action="store_true", help="禁用剔除机构")
    parser.add_argument("--test-ejector", action="store_true", help="启动前测试剔除机构")
    args = parser.parse_args()
    
    try:
        # 初始化系统
        initialize_system(args.config)
        
        # 测试剔除机构（如果需要）
        if args.test_ejector:
            logger.info("测试剔除机构...")
            io_controller.activate_rejector(0.5)
            time.sleep(1)
        
        # 启动传送带
        io_controller.start_conveyor()
        logger.info("传送带已启动，系统运行中...")
        
        # 运行直到被中断
        try:
            while running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("接收到中断信号，系统正在停止...")
        
    except Exception as e:
        logger.exception(f"系统运行过程中发生错误: {str(e)}")
    finally:
        # 清理资源
        cleanup()

if __name__ == "__main__":
    main()