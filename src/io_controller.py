#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版Jetson I/O控制模块

核心功能:
1. 位置传感器检测（基于中断）
2. 传送带控制（启动/停止）
3. 剔除执行器控制
"""

import os
import time
import logging
import threading
from typing import Callable, List
from enum import Enum

# 配置日志
logger = logging.getLogger("DefectInspection.IO")

# 尝试导入Jetson GPIO
try:
    import Jetson.GPIO as GPIO
    USE_JETSON_GPIO = True
    logger.info("使用Jetson.GPIO进行I/O控制")
except ImportError:
    USE_JETSON_GPIO = False
    logger.warning("Jetson.GPIO未安装，使用模拟模式")

class IOMode(Enum):
    """I/O模式枚举"""
    GPIO = 1      # 直接使用Jetson GPIO
    SIMULATE = 2  # 模拟模式，用于测试

class IOController:
    """Jetson I/O控制器"""
    
    def __init__(self, mode: IOMode = None, 
                 position_sensor_pin: int = 18,
                 conveyor_control_pin: int = 23,
                 rejector_control_pin: int = 24):
        """
        初始化I/O控制器
        
        Args:
            mode: I/O模式，如未指定则自动选择
            position_sensor_pin: 位置传感器GPIO引脚
            conveyor_control_pin: 传送带控制GPIO引脚
            rejector_control_pin: 执行器控制GPIO引脚
        """
        # 自动选择模式
        if mode is None:
            if USE_JETSON_GPIO:
                self.mode = IOMode.GPIO
            else:
                self.mode = IOMode.SIMULATE
        else:
            self.mode = mode
        
        # 引脚定义
        self.position_sensor_pin = position_sensor_pin
        self.conveyor_control_pin = conveyor_control_pin
        self.rejector_control_pin = rejector_control_pin
        
        # 状态变量
        self.initialized = False
        self.is_conveyor_running = False
        self.last_position_trigger = 0
        
        # 位置传感器回调
        self.position_callbacks = []
        self.running = False
        
        # 初始化I/O
        self._initialize_io()
        
    def _initialize_io(self):
        """初始化I/O系统"""
        try:
            if self.mode == IOMode.GPIO:
                if not USE_JETSON_GPIO:
                    logger.error("无法使用GPIO模式: Jetson.GPIO不可用")
                    self.mode = IOMode.SIMULATE
                else:
                    # 设置GPIO模式
                    GPIO.setmode(GPIO.BCM)
                    
                    # 配置位置传感器为输入，带内部上拉
                    GPIO.setup(self.position_sensor_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                    
                    # 配置传送带和执行器为输出，初始低电平
                    GPIO.setup(self.conveyor_control_pin, GPIO.OUT, initial=GPIO.LOW)
                    GPIO.setup(self.rejector_control_pin, GPIO.OUT, initial=GPIO.LOW)
                    
                    # 为位置传感器添加中断检测（下降沿触发）
                    GPIO.add_event_detect(self.position_sensor_pin, GPIO.FALLING, 
                                         callback=self._position_sensor_callback, bouncetime=200)
                    
                    logger.info("GPIO初始化完成")
                    
            elif self.mode == IOMode.SIMULATE:
                # 启动位置传感器模拟线程
                self.running = True
                self.position_thread = threading.Thread(
                    target=self._simulate_position_sensor,
                    daemon=True
                )
                self.position_thread.start()
                
                logger.info("模拟I/O模式已启动")
            
            # 标记初始化完成
            self.initialized = True
            
        except Exception as e:
            logger.error(f"I/O初始化失败: {e}")
            self.initialized = False
    
    def _position_sensor_callback(self, channel):
        """
        位置传感器中断回调函数
        
        Args:
            channel: 触发中断的GPIO通道
        """
        if not self.initialized:
            return
            
        # 防抖处理
        current_time = time.time()
        if current_time - self.last_position_trigger < 0.1:
            return
            
        self.last_position_trigger = current_time
        
        logger.debug(f"位置传感器触发 (引脚 {channel})")
        
        # 执行所有注册的回调
        for callback in self.position_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"位置传感器回调执行错误: {e}")
    
    def _simulate_position_sensor(self):
        """模拟位置传感器触发"""
        while self.running:
            # 仅在传送带运行时模拟传感器触发
            if self.is_conveyor_running:
                # 触发回调
                for callback in self.position_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"位置传感器回调执行错误: {e}")
                
                # 每2-3秒触发一次
                time.sleep(2 + (time.time() % 1))
            else:
                # 传送带停止时，减少轮询频率
                time.sleep(0.5)
    
    def register_position_callback(self, callback: Callable):
        """
        注册位置传感器回调函数
        
        Args:
            callback: 传感器触发时调用的函数
        """
        if callback not in self.position_callbacks:
            self.position_callbacks.append(callback)
            logger.debug(f"已注册位置传感器回调 (共 {len(self.position_callbacks)} 个)")
    
    def start_conveyor(self):
        """启动传送带"""
        if not self.initialized:
            logger.error("I/O未初始化，无法启动传送带")
            return False
        
        try:
            if self.mode == IOMode.GPIO:
                GPIO.output(self.conveyor_control_pin, GPIO.HIGH)
            
            self.is_conveyor_running = True
            logger.info("传送带已启动")
            return True
            
        except Exception as e:
            logger.error(f"启动传送带失败: {e}")
            return False
    
    def stop_conveyor(self):
        """停止传送带"""
        if not self.initialized:
            logger.error("I/O未初始化，无法停止传送带")
            return False
        
        try:
            if self.mode == IOMode.GPIO:
                GPIO.output(self.conveyor_control_pin, GPIO.LOW)
            
            self.is_conveyor_running = False
            logger.info("传送带已停止")
            return True
            
        except Exception as e:
            logger.error(f"停止传送带失败: {e}")
            return False
    
    def activate_rejector(self, duration: float = 0.5):
        """
        激活剔除执行器
        
        Args:
            duration: 激活持续时间(秒)
        """
        if not self.initialized:
            logger.error("I/O未初始化，无法激活执行器")
            return False
        
        try:
            if self.mode == IOMode.GPIO:
                # 激活执行器
                GPIO.output(self.rejector_control_pin, GPIO.HIGH)
                
                # 使用定时器在指定时间后关闭
                threading.Timer(duration, 
                               lambda: GPIO.output(self.rejector_control_pin, GPIO.LOW)
                              ).start()
            
            logger.info(f"执行器已激活 (持续 {duration} 秒)")
            return True
            
        except Exception as e:
            logger.error(f"激活执行器失败: {e}")
            return False
    
    def close(self):
        """清理I/O资源"""
        try:
            # 停止模拟线程
            self.running = False
            
            # 停止传送带
            if self.is_conveyor_running:
                self.stop_conveyor()
            
            # 清理GPIO资源
            if self.mode == IOMode.GPIO and USE_JETSON_GPIO:
                GPIO.cleanup()
            
            logger.info("I/O控制器已关闭")
            
        except Exception as e:
            logger.error(f"关闭I/O资源时出错: {e}")
        finally:
            self.initialized = False

# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 检查是否在Jetson上运行
    on_jetson = os.path.exists('/proc/device-tree/model') and 'NVIDIA' in open('/proc/device-tree/model').read()
    io_mode = IOMode.GPIO if on_jetson else IOMode.SIMULATE
    
    # 创建I/O控制器
    io = IOController(mode=io_mode)
    
    # 注册位置传感器回调
    def position_callback():
        logger.info("检测到产品！")
        io.stop_conveyor()
        # 模拟检测过程
        time.sleep(1)
        # 模拟剔除不良品
        io.activate_rejector(0.5)
        # 重启传送带
        time.sleep(0.5)
        io.start_conveyor()
    
    io.register_position_callback(position_callback)
    
    try:
        # 启动传送带
        logger.info("启动传送带...")
        io.start_conveyor()
        
        # 运行一段时间
        logger.info("系统运行中，按Ctrl+C退出...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在退出...")
    finally:
        # 清理资源
        io.close()