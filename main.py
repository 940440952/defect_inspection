#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
瑕疵检测系统主程序 - 连续传送带版本

系统流程:
1. 拍照位置传感器检测到产品
2. 摄像头捕获产品图像（不停止传送带）
3. 图像加入处理队列
4. 处理线程进行瑕疵检测
5. 检测结果和图像上传到服务器
6. 如有瑕疵，产品ID加入剔除队列
7. 剔除位置传感器检测到产品
8. 如果该产品在剔除队列中，激活剔除机构
"""

import os
import sys
import time
import json
import argparse
import logging
import threading
import signal
import cv2
import queue
from datetime import datetime
from typing import Dict, Any, List, Optional

# 导入系统模块
from src.io_controller import PipelineController
from src.detector import YOLODetector as DefectDetector
from src.cropper import ImageCropper
from src.display import DefectDisplay
from src.api_client import APIClient
from src.utils import setup_logging, load_config, ensure_directory_exists, convert_grayscale_to_rgb
from src.product_tracker import ProductTracker

# 全局变量
running = True
io_controller = None
camera = None
detector = None
cropper = None
api_client = None
display = None
logger = None
product_tracker = None
config = None
args = None

# 添加UI消息队列
ui_message_queue = queue.Queue()

# 线程对象
processing_thread = None
ejection_thread = None

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="瑕疵检测系统")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--log-level", type=str, default="info", 
                        choices=["debug", "info", "warning", "error"],
                        help="日志级别 (debug, info, warning, error)")
    parser.add_argument("--no-display", action="store_true", help="禁用结果显示界面")
    parser.add_argument("--no-ejection", action="store_true", help="禁用剔除机构")
    parser.add_argument("--test-pipeline", action="store_true", help="测试流水线运行")
    parser.add_argument("--camera-index", type=int, help="指定摄像头索引，覆盖配置文件")
    parser.add_argument("--save-dir", type=str, help="指定检测结果保存目录，覆盖配置文件")
    return parser.parse_args()


def setup_signal_handlers():
    """设置信号处理，以便优雅地关闭程序"""
    def signal_handler(sig, frame):
        global running
        logger.info("接收到终止信号，准备关闭系统...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def initialize_system(config: Dict[str, Any], args):
    """初始化系统各组件"""
    global io_controller, camera, detector, api_client, display, logger, cropper, product_tracker

    logger.info("正在初始化系统组件...")
    
    # 初始化产品跟踪器
    product_tracker = ProductTracker(max_products=100)
    logger.info("产品跟踪器初始化成功")
    
    # 1. 初始化IO控制器
    try:
        io_config = config.get("io_controller", {})
        io_controller = PipelineController(
            photo_sensor_pin=io_config.get("photo_sensor_pin", 7),
            conveyor_pin=io_config.get("conveyor_pin", 29),
            rejector_pin=io_config.get("rejector_pin", 31),
            ejection_sensor_pin=io_config.get("ejection_sensor_pin",32),
            photo_sensor_register=io_config.get("position_register", "0x02448030 w 0x040"),
            conveyor_register=io_config.get("conveyor_register", "0x02430068 w 0x004"),
            rejector_register=io_config.get("rejector_register", "0x02430070 w 0x004"),
            ejection_sensor_register=config.get("ejection_sensor_register","0x02434080 w 0x040")
        )
        logger.info("IO控制器初始化成功")
        
    except Exception as e:
        logger.error(f"IO控制器初始化失败: {str(e)}")
        return False
    
    # 2. 初始化摄像头
    try:
        from src.camera import CameraManager
        camera_config = config.get("camera", {})
        # 如果命令行指定了摄像头索引，则覆盖配置文件
        if args.camera_index is not None:
            camera_index = args.camera_index
        else:
            camera_index = camera_config.get("index", 0)
        
        camera = CameraManager(device_index=camera_index)
        # 尝试打开摄像头进行验证
        if not camera.open_camera():
            logger.error("摄像头打开失败")
            return False
            
        logger.info(f"摄像头 #{camera_index} 初始化成功")
    except Exception as e:
        logger.error(f"摄像头初始化失败: {str(e)}")
        return False
    
    # 3. 初始化裁剪器
    try:
        cropper_config = config.get("cropper", {})
        cropper = ImageCropper(
            model_name=cropper_config.get("cropper_model_name","yolo11l"),
            conf_threshold=cropper_config.get("cropper_confidence_threshold", 0.25),
            nms_threshold=cropper_config.get("nms_threshold", 0.45),
            models_dir=cropper_config.get("models_dir", "/home/gtm/defect_inspection/models/cropper"),
        )
        logger.info("裁剪器初始化成功")
    except Exception as e:
        logger.error(f"裁剪器初始化失败: {str(e)}")
        return False

    # 4. 初始化检测模型
    try:
        detector_config = config.get("detector", {})
        # 获取检测器过滤配置
        filter_config = detector_config.get("filter", {})

        detector = DefectDetector(
            model_name=detector_config.get("model_name", "yolo11l"),
            models_dir=detector_config.get("models_dir", "/home/gtm/defect_inspection/models/detector"),
            filter_enabled=filter_config.get("enabled", True),
            min_area=filter_config.get("min_area", 100),
            confidence_threshold=filter_config.get("confidence_threshold", 0.25)
        )
            
        # 设置类别名称
        class_names = detector_config.get("class_names", ["defect"])
        detector.class_names = class_names
        logger.info("检测模型初始化成功")
    except Exception as e:
        logger.error(f"检测模型初始化失败: {str(e)}")
        return False
    
    # 5. 初始化API客户端
    try:
        api_config = config.get("api", {})

        api_client = APIClient(
            api_url=api_config.get("api_url", ""),
            api_token=api_config.get("auth_token", ""),
            line_name=api_config.get("line_name", "LineTest"),
            product_type=api_config.get("product_type", "QC"),
            timeout=api_config.get("timeout", 10),
            max_retries=api_config.get("max_retries", 3),
            retry_delay=api_config.get("retry_delay", 5)
        )

        if api_config.get("api_url") and api_config.get("auth_token"):
            logger.info("API客户端初始化成功")
        else:
            logger.warning("API参数不完整，API功能将不可用")
    except Exception as e:
        logger.error(f"API客户端初始化失败: {str(e)}")
        # API客户端失败不阻止系统运行
        logger.warning("继续运行系统，但无法上传检测结果")
        api_client = None
    
    # 6. 初始化显示界面(可选)
    try:
        if config.get("display", {}).get("enabled", True) and not args.no_display:
            display_config = config.get("display", {})
            # 从detector配置中获取类别名称
            class_names = detector.class_names if detector else []
            
            display = DefectDisplay(
                window_title=display_config.get("window_name", "瑕疵检测系统"),
                class_names=class_names  # 传递类别名称到显示界面
            )
            logger.info("显示界面初始化成功")
        else:
            display = None
            logger.info("显示界面已禁用")
    except Exception as e:
        logger.error(f"显示界面初始化失败: {str(e)}")
        return False
    
    # 7. 测试流水线(可选)
    if args.test_pipeline:
        logger.info("开始测试流水线...")
        try:
            # 启动传送带
            io_controller.start_conveyor()
            time.sleep(2)
            # 停止传送带
            io_controller.stop_conveyor()
            time.sleep(1)
            # 测试剔除机构
            if not args.no_ejection:
                io_controller.activate_rejector(0.5)
                logger.info("剔除机构测试完成")
                time.sleep(1)
            logger.info("流水线测试成功")
        except Exception as e:
            logger.error(f"流水线测试失败: {str(e)}")
            return False
    
    return True


def cleanup_system():
    """清理系统资源"""
    global processing_thread, ejection_thread
    
    logger.info("正在清理系统资源...")
    
    # 停止所有线程
    if processing_thread and processing_thread.is_alive():
        logger.info("等待处理线程结束...")
        processing_thread.join(timeout=5.0)
        
    if ejection_thread and ejection_thread.is_alive():
        logger.info("等待剔除线程结束...")
        ejection_thread.join(timeout=5.0)
    
    # 显示系统运行统计
    if product_tracker:
        stats = product_tracker.stats  # 直接访问 stats 属性
        logger.info(f"系统运行统计:")
        logger.info(f"- 检测产品总数: {stats['total_products']}")
        logger.info(f"- 检测到的瑕疵产品数: {stats['defect_products']}")
        logger.info(f"- 已剔除产品数: {stats['ejected_products']}")
        logger.info(f"- 已处理产品数: {stats['processed_products']}")
        if stats['total_products'] > 0:
            defect_rate = stats['defect_products']/max(1, stats['total_products'])*100
            logger.info(f"- 瑕疵率: {defect_rate:.1f}%")
    
    # 关闭IO控制器
    if io_controller:
        try:
            io_controller.close()
        except Exception as e:
            logger.error(f"关闭IO控制器失败: {str(e)}")
    
    # 释放摄像头
    if camera:
        try:
            camera.close_camera()
        except Exception as e:
            logger.error(f"释放摄像头资源失败: {str(e)}")
    
    # 关闭显示界面
    if display:
        try:
            display.close()
        except Exception as e:
            logger.error(f"关闭显示界面失败: {str(e)}")
    
    logger.info("系统资源清理完成")


def update_ui(message_type, *args):
    """安全地将UI更新请求添加到消息队列"""
    if display:
        ui_message_queue.put((message_type, args))
        return True
    return False


def photo_sensor_callback():
    """拍照位置传感器触发的回调函数"""
    if not running:
        return
        
    try:
        # 记录产品进入系统
        product_id = product_tracker.photo_sensor_triggered()
        
        # 立即拍照，不停止传送带
        update_ui("set_status", f"捕获产品 {product_id} 图像...", False)
        
        # 捕获图像
        image, _ = camera.capture_image(return_array=True)
        
        if image is None:
            logger.error(f"产品 {product_id} 图像捕获失败")
            update_ui("set_status", f"产品 {product_id} 图像捕获失败", True)
            return
            
        # 转换为RGB图像 (YOLO使用RGB格式)
        image = convert_grayscale_to_rgb(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 将图像设置到产品并加入处理队列
        product_tracker.set_product_image(product_id, image)
        
        logger.info(f"产品 {product_id} 图像已捕获，加入处理队列")
        
    except Exception as e:
        logger.error(f"拍照传感器回调执行出错: {str(e)}")


def ejection_sensor_callback():
    """剔除位置传感器触发的回调函数"""
    if not running:
        return
        
    try:
        # 查询是否需要剔除此产品
        product_id = product_tracker.ejection_sensor_triggered()
        
        # 如果需要剔除
        if product_id is not None:
            # 执行剔除操作
            ejection_config = config.get("ejection", {})
            if ejection_config.get("enabled", True) and not args.no_ejection:
                logger.info(f"剔除产品 {product_id}")
                update_ui("set_status", f"剔除瑕疵产品 {product_id}...", False)
                
                # 获取剔除时间
                duration = ejection_config.get("duration", 0.5)
                
                # 激活剔除机构
                io_controller.activate_rejector(duration)
                
            else:
                logger.info(f"产品 {product_id} 需要剔除，但剔除功能已禁用")
                update_ui("set_status", f"产品 {product_id} 需要剔除 (已禁用)", False)
        else:
            logger.debug("产品通过检测，无需剔除")
            
    except Exception as e:
        logger.error(f"剔除传感器回调执行出错: {str(e)}")

def processing_thread_function():
    """处理线程函数 - 处理队列中的图像"""
    logger.info("处理线程已启动")
    
    while running:
        try:
            # 等待新产品加入处理队列，使用事件通知机制
            if not product_tracker.wait_for_new_product(timeout=1.0):
                # 超时继续循环，检查running状态
                continue
                
            # 获取待处理产品ID
            while running:  # 处理队列中所有产品
                product_id = product_tracker.get_next_product_for_processing()
                if product_id is None:
                    # 队列已清空，等待新的事件
                    break
                    
                # 获取产品信息
                product = product_tracker.find_product_by_id(product_id)
                if product is None or product.image is None:
                    logger.warning(f"找不到产品 {product_id} 或图像为空")
                    continue
                    
                logger.info(f"开始处理产品 {product_id}")
                update_ui("set_status", f"处理产品 {product_id}...", False)
                
                # 处理图像
                image = product.image
                
                # 1. 首先进行裁剪检测
                cropped_results = cropper.detect_and_crop(image)
                
                # 初始化检测结果变量
                all_detections = []
                crop_detections = []
                has_defect = False
                
                # 2. 如果裁剪检测到缺陷区域
                if cropped_results:
                    logger.info(f"产品 {product_id} 裁剪器检测到{len(cropped_results)}个潜在区域")
                    
                    # 对裁剪的区域进行详细检测
                    for crop_idx, crop_result in enumerate(cropped_results):
                        crop_img = crop_result['crop']
                        crop_box = crop_result['box']
                        
                        # 对裁剪图像进行检测
                        detections = detector.detect(crop_img)
                        
                        # 保存原始检测结果
                        crop_detections.append(detections)
                        
                        # 如果有检测结果
                        if detections:
                            # 标记产品有缺陷
                            has_defect = True
                            
                            # 调整检测坐标到原始图像坐标系
                            for det in detections:
                                # 原始检测格式: [x1, y1, x2, y2, conf, class_id]
                                # 调整坐标: 加上裁剪区域的左上角坐标
                                x1, y1 = det[0] + crop_box[0], det[1] + crop_box[1]
                                x2, y2 = det[2] + crop_box[0], det[3] + crop_box[1]
                                all_detections.append([x1, y1, x2, y2, det[4], det[5]])
                
                # 3. 更新产品检测结果
                product_tracker.update_product_detection(
                    product_id, all_detections, has_defect, cropped_results
                )
                
                # 4. 如果有显示界面，更新显示
                if display:
                    # 整合检测结果用于显示
                    from src.display_utils import draw_combined_detections, merge_detection_results
                    
                    # 如果有裁剪结果，使用增强版绘制
                    if cropped_results:
                        # 将检测结果组织为字典
                        detection_dict = merge_detection_results(cropped_results, crop_detections)
                        
                        # 使用增强版绘制函数
                        display_img = draw_combined_detections(
                            image, 
                            cropped_results, 
                            detection_dict,
                            class_names=detector.class_names,
                            show_labels=True,
                            show_confidence=True
                        )
                    else:
                        # 否则使用原有绘制函数
                        display_img = detector.draw_detections(image, all_detections)
                        
                    # 更新UI
                    update_ui("update_image", display_img, all_detections)
                
                # 5. 上传服务器 (如果配置了API客户端)
                if api_client and api_client.api_url and api_client.api_token:
                    try:
                        logger.info(f"上传产品 {product_id} 检测结果到服务器")
                        update_ui("set_status", f"上传产品 {product_id} 结果...", False)
                        
                        # 直接传递图像和检测结果
                        api_client.import_to_label_studio(
                            image, 
                            all_detections, 
                            {"product_id": product_id}, 
                            is_path=False
                        )
                        
                        logger.info(f"产品 {product_id} 结果上传成功")
                    except Exception as e:
                        logger.error(f"产品 {product_id} 结果上传失败: {str(e)}")
                
                logger.info(f"产品 {product_id} 处理完成")
            
        except Exception as e:
            logger.error(f"处理线程异常: {str(e)}")
            time.sleep(1)  # 出错后等待一段时间再继续
    
    logger.info("处理线程已退出")

def process_ui_messages():
    """处理UI消息队列中的消息"""
    if not display or not hasattr(display, 'root'):
        return
    
    try:
        while not ui_message_queue.empty():
            message_type, args = ui_message_queue.get_nowait()
            
            if message_type == "set_status":
                display.set_status(args[0], args[1])
            elif message_type == "update_image":
                display.update_image(args[0], args[1])
            
            ui_message_queue.task_done()
    except Exception as e:
        logger.error(f"处理UI消息出错: {str(e)}")
    finally:
        # 重新安排此函数执行
        if display and hasattr(display, 'root') and display.running:
            display.root.after(100, process_ui_messages)


def main():
    """主函数"""
    global running, display, logger, args, config, processing_thread, ejection_thread
    
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志级别
    log_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR
    }
    log_level = log_level_map.get(args.log_level.lower(), logging.INFO)
    
    # 初始化日志系统
    logger = setup_logging(level=log_level, log_file="logs/defect_inspection.log")
    logger.info("----- 瑕疵检测系统启动 (连续传送带模式) -----")
    
    # 设置信号处理
    setup_signal_handlers()
    
    try:
        # 加载配置
        config = load_config(args.config)
        logger.info(f"成功加载配置文件: {args.config}")
        
        # 初始化系统
        if not initialize_system(config, args):
            logger.error("系统初始化失败，程序终止")
            return 1
        
        logger.info("系统初始化完成，准备开始检测")
        
        # 注册传感器回调
        io_controller.register_photo_callback(photo_sensor_callback)
                
        # 注册剔除传感器回调
        io_controller.register_ejection_callback(ejection_sensor_callback)
        
        processing_thread = threading.Thread(
            target=processing_thread_function, 
            name="ProcessingThread", 
            daemon=True
        )
        processing_thread.start()
        
        # 启动传送带
        logger.info("启动传送带，开始检测流程")
        io_controller.start_conveyor()
        
        # 根据是否使用显示界面，选择不同的主循环
        if display:
            # 在主线程中启动和运行显示界面
            display.set_status("系统就绪，开始检测")
            display.start()  # 设置界面状态为运行中
            
            # 设置UI消息处理
            display.root.after(100, process_ui_messages)
            
            # 在主线程中运行GUI主循环 (这会阻塞，直到窗口关闭)
            display.run()
            
            # GUI关闭后，标记程序结束
            running = False
        else:
            # 无GUI模式 - 使用简单的循环
            logger.info("系统运行中，按Ctrl+C退出...")
            
            # 主循环 - 保持程序运行并处理信号
            while running:
                time.sleep(0.5)  # 短暂睡眠避免CPU占用过高
    
    except KeyboardInterrupt:
        logger.info("用户中断，准备关闭系统")
        running = False
    
    except Exception as e:
        logger.error(f"系统运行时发生异常: {str(e)}")
        if display:
            try:
                display.set_status(f"系统错误: {str(e)}", is_error=True)
            except:
                pass  # 忽略可能的线程问题
    
    finally:
        # 清理资源
        cleanup_system()
        logger.info("----- 瑕疵检测系统已关闭 -----")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())