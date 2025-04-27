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
import json
import argparse
import logging
import threading
import signal
import cv2

import queue
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.utils import convert_grayscale_to_rgb

# 导入系统模块
from src.io_controller import PipelineController
from src.detector import YOLODetector as DefectDetector
from src.display import DefectDisplay
from src.api_client import APIClient
from src.utils import setup_logging, load_config, ensure_directory_exists


# 全局变量
running = True
io_controller = None
camera = None
detector = None
api_client = None
display = None
logger = None
# 添加处理锁，防止重入
processing_lock = threading.Lock()
# 添加处理标志，表示是否有产品正在处理中
processing_active = False
# 添加UI消息队列
ui_message_queue = queue.Queue()

# 记录检测统计
stats = {
    "total_inspected": 0,
    "defects_detected": 0,
    "defect_types": {},
    "system_start_time": None
}

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
    global io_controller, camera, detector, api_client, display, stats, logger
    
    logger.info("正在初始化系统组件...")
    stats["system_start_time"] = datetime.now()
    
    # 初始化统计数据中的缺陷类型计数
    stats["defect_types"] = {"defect": 0}
    
    # 1. 初始化IO控制器
    try:
        io_config = config.get("io_controller", {})
        io_controller = PipelineController(
            position_sensor_pin=io_config.get("position_sensor_pin", 7),
            conveyor_pin=io_config.get("conveyor_pin", 29),
            rejector_pin=io_config.get("rejector_pin", 31),
            position_register=io_config.get("position_register", "0x02448030 w 0x040"),
            conveyor_register=io_config.get("conveyor_register", "0x02430068 w 0x004"),
            rejector_register=io_config.get("rejector_register", "0x02430070 w 0x004")
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
    
    # 3. 初始化检测模型
    try:
        detector_config = config.get("detector", {})
        detector = DefectDetector(
            model_name=detector_config.get("model_name", "yolo11l"),
            conf_thresh=detector_config.get("confidence_threshold", 0.25),
            nms_thresh=detector_config.get("nms_threshold", 0.45),
            models_dir=detector_config.get("models_dir", "models"),
            use_dla=detector_config.get("use_dla", True)
        )
        # 设置类别名称为只有一种缺陷
        detector.class_names = detector_config.get("class_names", ["defect"])
        logger.info("检测模型初始化成功")
    except Exception as e:
        logger.error(f"检测模型初始化失败: {str(e)}")
        return False
    
    # 4. 初始化API客户端
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
    
    # 5. 初始化显示界面(可选)
    if not args.no_display and config.get("display", {}).get("enabled", True):
        try:
            display_config = config.get("display", {})
            display = DefectDisplay(display_config.get("window_name", "瑕疵检测系统"))
            logger.info("显示界面初始化成功")
        except Exception as e:
            logger.error(f"显示界面初始化失败: {str(e)}")
            # 显示界面失败不阻止系统运行
            display = None
    
    # 6. 测试流水线(可选)
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
    logger.info("正在清理系统资源...")
    
    # 显示系统运行统计
    if stats["system_start_time"]:
        runtime = (datetime.now() - stats["system_start_time"]).total_seconds() / 60.0
        logger.info(f"系统运行统计:")
        logger.info(f"- 运行时间: {runtime:.1f}分钟")
        logger.info(f"- 检测产品总数: {stats['total_inspected']}")
        logger.info(f"- 检测到的瑕疵产品数: {stats['defects_detected']}")
        if stats['total_inspected'] > 0:
            logger.info(f"- 瑕疵率: {stats['defects_detected']/max(1, stats['total_inspected'])*100:.1f}%")
        
        # 输出各类型的瑕疵统计
        logger.info("- 各类型瑕疵统计:")
        for defect_type, count in stats["defect_types"].items():
            logger.info(f"  - {defect_type}: {count}")
    
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


def save_detection_result(image, detection_results, save_dir):
    """
    保存检测图像和结果 (仅在需要时使用)
    此功能默认被禁用，除非显式配置
    """
    # 检查是否配置了输出目录
    if not save_dir:
        return None, None
        
    try:
        # 确保保存目录存在
        ensure_directory_exists(save_dir)
        
        # 生成时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        image_path = os.path.join(save_dir, f"defect_{timestamp}.jpg")
        
        # 保存原始图像
        cv2.imwrite(image_path, image)
        
        # 保存检测结果
        result_path = os.path.join(save_dir, f"defect_{timestamp}.json")
        with open(result_path, 'w') as f:
            json.dump(detection_results, f)
            
        logger.debug(f"保存检测结果到: {image_path}, {result_path}")
        return image_path, result_path
    except Exception as e:
        logger.error(f"保存检测结果失败: {str(e)}")
        return None, None


# 新增：更新UI的函数
def update_ui(message_type, *args):
    """安全地将UI更新请求添加到消息队列"""
    if display:
        ui_message_queue.put((message_type, args))
        return True
    return False


def process_product(stop_conveyor=True):
    """处理一个产品的检测流程"""
    global stats, processing_active
    
    try:
        # 1. 停止传送带
        if stop_conveyor:
            logger.info("产品到达检测位置，停止传送带")
            update_ui("set_status", "产品到达，准备检测...", False)
            io_controller.stop_conveyor()
        
        # 停止传送带后等待一段时间，确保产品静止
        time.sleep(0.5)

        # 2. 拍摄图像
        logger.info("捕获产品图像")
        update_ui("set_status", "捕获产品图像...", False)
            
        # 捕获图像
        image, _ = camera.capture_image(return_array=False, save_path="test_image.jpg")
        
        if image is None:
            logger.error("图像捕获失败")
            update_ui("set_status", "图像捕获失败", True)
            io_controller.start_conveyor()  # 重启传送带继续生产
            return
            
        # 转换为3通道图像
        image = convert_grayscale_to_rgb(image)
        # numpy.ndarray转换为yolo能识别的图像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. 运行检测
        logger.info("执行瑕疵检测")
        update_ui("set_status", "执行瑕疵检测...", False)
        detection_results = detector.detect(image)
        
        # 更新统计
        stats["total_inspected"] += 1
        
        # 检测到缺陷
        has_defect = len(detection_results) > 0
        if has_defect:
            stats["defects_detected"] += 1
            logger.info(f"检测到{len(detection_results)}个瑕疵")
            
            # 更新缺陷统计 - 简化为只计数总数
            stats["defect_types"]["defect"] += len(detection_results)
            update_ui("set_status", f"检测到{len(detection_results)}个瑕疵", False)
        else:
            logger.info("未检测到瑕疵")
            update_ui("set_status", "产品合格，未检测到瑕疵", False)

        # 更新显示界面
        # 在图像上绘制检测结果
        display_img = detector.draw_detections(image, detection_results)
        update_ui("update_image", display_img, detection_results)
        
        # 4. 上传服务器 (如果配置了API客户端)
        if api_client and api_client.api_url and api_client.api_token:
            logger.info("上传检测结果到服务器")
            update_ui("set_status", "上传检测结果到服务器...", False)
            
            try:
                # 直接传递图像和检测结果，不保存到本地
                api_client.import_to_label_studio(image, detection_results, is_path=False)
                logger.info("结果上传成功")
            except Exception as e:
                logger.error(f"结果上传失败: {str(e)}")
                update_ui("set_status", "结果上传失败", True)
        
        # 5. 处理缺陷产品
        ejection_config = config.get("ejection", {})
        if has_defect and ejection_config.get("enabled", True) and not args.no_ejection:
            logger.info("启动剔除机构处理缺陷产品")
            update_ui("set_status", "启动剔除机构...", False)
            duration = ejection_config.get("duration", 0.5)
            io_controller.activate_rejector(duration)
        
        # 6. 重启传送带
        logger.info("重启传送带，准备下一个产品检测")
        update_ui("set_status", "等待下一个产品...", False)
        io_controller.start_conveyor()
        
    except Exception as e:
        logger.error(f"产品检测过程出现异常: {str(e)}")
        update_ui("set_status", f"系统错误: {str(e)}", True)
        # 确保传送带重新启动
        try:
            io_controller.start_conveyor()
        except:
            pass
    finally:
        # 释放处理标志
        with processing_lock:
            processing_active = False


def position_sensor_callback():
    """位置传感器触发的回调函数"""
    global processing_lock, processing_active
    
    if not running:
        return
        
    # 检查是否已有处理线程在运行
    with processing_lock:
        if processing_active:
            logger.warning("已有产品正在处理中，忽略此次触发")
            return
        processing_active = True
    
    logger.info("位置传感器检测到产品到达")
    # 启动检测流程，使用线程避免阻塞位置传感器处理
    threading.Thread(target=process_product, name="ProductProcessing").start()


# 处理UI消息队列的函数
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
    global running, display, logger, args, config
    
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
    logger.info("----- 瑕疵检测系统启动 -----")
    
    # 设置信号处理
    setup_signal_handlers()
    
    try:
        # 加载配置
        config = load_config(args.config)
        # 打印
        logger.debug(f"加载的配置: {json.dumps(config, indent=2)}")
        
        logger.info(f"成功加载配置文件: {args.config}")
        
        # 初始化系统
        if not initialize_system(config, args):
            logger.error("系统初始化失败，程序终止")
            return 1
        
        logger.info("系统初始化完成，开始检测")
        
        # 根据是否使用显示界面，选择不同的主循环
        if display:
            # 在主线程中启动和运行显示界面
            display.set_status("系统就绪，准备开始检测")
            display.start()  # 设置界面状态为运行中
            
            # 注册位置传感器回调
            io_controller.register_position_callback(position_sensor_callback)
            
            # 启动传送带 - 在工作线程中，避免阻塞GUI
            def start_conveyor_thread():
                logger.info("启动传送带，开始检测流程")
                io_controller.start_conveyor()
            
            threading.Thread(target=start_conveyor_thread, daemon=True).start()
            
            # 设置UI消息处理
            display.root.after(100, process_ui_messages)
            
            # 在主线程中运行GUI主循环 (这会阻塞，直到窗口关闭)
            display.run()
            
            # GUI关闭后，标记程序结束
            running = False
        else:
            # 无GUI模式 - 使用简单的循环
            # 注册位置传感器回调
            io_controller.register_position_callback(position_sensor_callback)
            
            # 启动传送带
            logger.info("启动传送带，开始检测流程")
            io_controller.start_conveyor()
            
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