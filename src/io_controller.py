#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson I/O Control Module

用于 Jetson 与外部传感器、执行器的交互:
1. 位置传感器 IO 信号接收
2. 传送带 IO 控制 (启动/暂停)
3. 筛除执行器 IO 控制

支持通过 GPIO 或工业 I/O 模块进行通信
"""

import os
import time
import logging
import threading
from typing import Callable, Dict, Optional, List, Union
from enum import Enum

# 配置日志
logger = logging.getLogger("DefectInspection.IO")

# 尝试导入 Jetson GPIO
try:
    import Jetson.GPIO as GPIO
    USE_JETSON_GPIO = True
    logger.info("使用 Jetson.GPIO 进行 I/O 控制")
except ImportError:
    USE_JETSON_GPIO = False
    logger.warning("Jetson.GPIO 未安装，使用模拟模式")

class IOMode(Enum):
    """I/O 模式枚举"""
    GPIO = 1  # 直接使用 Jetson GPIO
    MODBUS = 2  # 使用 Modbus RTU/TCP
    SIMULATE = 3  # 模拟模式，用于测试

class IOController:
    """Jetson I/O 控制器"""
    
    def __init__(self, mode: IOMode = None, 
                 position_sensor_pin: int = 18,
                 conveyor_control_pin: int = 23,
                 rejector_control_pin: int = 24,
                 modbus_device: str = "/dev/ttyUSB0",
                 modbus_baudrate: int = 9600):
        """
        初始化 I/O 控制器
        
        Args:
            mode: I/O 模式, 如未指定则自动选择
            position_sensor_pin: 位置传感器 GPIO 引脚
            conveyor_control_pin: 传送带控制 GPIO 引脚
            rejector_control_pin: 筛除执行器控制 GPIO 引脚
            modbus_device: Modbus 设备路径 (仅 MODBUS 模式)
            modbus_baudrate: Modbus 波特率 (仅 MODBUS 模式)
        """
        # 自动选择模式
        if mode is None:
            if USE_JETSON_GPIO:
                self.mode = IOMode.GPIO
            else:
                self.mode = IOMode.SIMULATE
        else:
            self.mode = mode
        
        # 配置引脚定义
        self.position_sensor_pin = position_sensor_pin
        self.conveyor_control_pin = conveyor_control_pin
        self.rejector_control_pin = rejector_control_pin
        
        # Modbus 配置
        self.modbus_device = modbus_device
        self.modbus_baudrate = modbus_baudrate
        self.modbus_client = None
        
        # 状态变量
        self.initialized = False
        self.is_conveyor_running = False
        self.last_position_trigger = 0
        
        # 位置传感器回调
        self.position_callbacks = []
        self.position_thread = None
        self.running = False
        
        # 初始化 I/O
        self._initialize_io()
        
    def _initialize_io(self):
        """初始化 I/O 系统"""
        try:
            if self.mode == IOMode.GPIO:
                if not USE_JETSON_GPIO:
                    logger.error("无法使用 GPIO 模式: Jetson.GPIO 不可用")
                    self.mode = IOMode.SIMULATE
                else:
                    # 设置 GPIO 模式
                    GPIO.setmode(GPIO.BCM)
                    
                    # 配置输入引脚 (位置传感器)
                    GPIO.setup(self.position_sensor_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                    
                    # 配置输出引脚 (传送带和筛除器)
                    GPIO.setup(self.conveyor_control_pin, GPIO.OUT, initial=GPIO.LOW)
                    GPIO.setup(self.rejector_control_pin, GPIO.OUT, initial=GPIO.LOW)
                    
                    # 为位置传感器添加事件检测
                    GPIO.add_event_detect(self.position_sensor_pin, GPIO.FALLING, 
                                         callback=self._position_sensor_callback, bouncetime=200)
                    
                    logger.info("GPIO 初始化完成")
                    
            elif self.mode == IOMode.MODBUS:
                # 导入 Modbus 库
                try:
                    import minimalmodbus
                    import serial
                    
                    # 创建 Modbus 客户端
                    self.modbus_client = minimalmodbus.Instrument(self.modbus_device, 1)
                    self.modbus_client.serial.baudrate = self.modbus_baudrate
                    self.modbus_client.serial.timeout = 0.5
                    
                    # 启动位置传感器监视线程
                    self.running = True
                    self.position_thread = threading.Thread(
                        target=self._poll_position_sensor,
                        daemon=True
                    )
                    self.position_thread.start()
                    
                    logger.info(f"Modbus 初始化完成 (设备: {self.modbus_device})")
                    
                except ImportError:
                    logger.error("无法使用 Modbus 模式: minimalmodbus 包未安装")
                    self.mode = IOMode.SIMULATE
                    
            elif self.mode == IOMode.SIMULATE:
                # 模拟模式不需要硬件初始化
                # 启动位置传感器模拟线程
                self.running = True
                self.position_thread = threading.Thread(
                    target=self._simulate_position_sensor,
                    daemon=True
                )
                self.position_thread.start()
                
                logger.info("模拟 I/O 模式已启动")
            
            # 标记初始化完成
            self.initialized = True
            
        except Exception as e:
            logger.error(f"I/O 初始化失败: {e}")
            self.initialized = False
    
    def _position_sensor_callback(self, channel):
        """
        位置传感器 GPIO 回调函数
        
        Args:
            channel: 触发回调的 GPIO 通道
        """
        if not self.initialized:
            return
            
        # 防止抖动
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
    
    def _poll_position_sensor(self):
        """Modbus 模式下轮询位置传感器"""
        last_state = False
        
        while self.running:
            try:
                # 读取位置传感器状态 (假设使用寄存器 0)
                current_state = bool(self.modbus_client.read_bit(0))
                
                # 检测下降沿 (传感器从高电平变为低电平)
                if last_state and not current_state:
                    # 触发回调
                    for callback in self.position_callbacks:
                        try:
                            callback()
                        except Exception as e:
                            logger.error(f"位置传感器回调执行错误: {e}")
                
                last_state = current_state
                
                # 短暂休眠
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Modbus 轮询错误: {e}")
                time.sleep(1)  # 出错后延迟重试
    
    def _simulate_position_sensor(self):
        """模拟位置传感器信号"""
        while self.running:
            # 仅在传送带运行时模拟传感器触发
            if self.is_conveyor_running:
                # 触发所有注册的回调
                for callback in self.position_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"位置传感器回调执行错误: {e}")
                
                # 每 2-4 秒触发一次传感器信号
                time.sleep(2 + 2 * (time.time() % 1))
            else:
                # 传送带停止时，减少轮询频率
                time.sleep(0.5)
    
    def register_position_callback(self, callback: Callable):
        """
        注册位置传感器回调函数
        
        Args:
            callback: 触发时调用的函数
        """
        if callback not in self.position_callbacks:
            self.position_callbacks.append(callback)
            logger.debug(f"已注册位置传感器回调 (共 {len(self.position_callbacks)} 个)")
    
    def start_conveyor(self):
        """启动传送带"""
        if not self.initialized:
            logger.error("I/O 未初始化，无法启动传送带")
            return False
        
        try:
            if self.mode == IOMode.GPIO:
                GPIO.output(self.conveyor_control_pin, GPIO.HIGH)
            elif self.mode == IOMode.MODBUS:
                # 假设使用线圈 0 控制传送带
                self.modbus_client.write_bit(0, 1)
            
            self.is_conveyor_running = True
            logger.info("传送带已启动")
            return True
            
        except Exception as e:
            logger.error(f"启动传送带失败: {e}")
            return False
    
    def stop_conveyor(self):
        """停止传送带"""
        if not self.initialized:
            logger.error("I/O 未初始化，无法停止传送带")
            return False
        
        try:
            if self.mode == IOMode.GPIO:
                GPIO.output(self.conveyor_control_pin, GPIO.LOW)
            elif self.mode == IOMode.MODBUS:
                # 假设使用线圈 0 控制传送带
                self.modbus_client.write_bit(0, 0)
            
            self.is_conveyor_running = False
            logger.info("传送带已停止")
            return True
            
        except Exception as e:
            logger.error(f"停止传送带失败: {e}")
            return False
    
    def toggle_conveyor(self):
        """切换传送带状态"""
        if self.is_conveyor_running:
            return self.stop_conveyor()
        else:
            return self.start_conveyor()
    
    def activate_rejector(self, duration: float = 0.5):
        """
        激活筛除执行器
        
        Args:
            duration: 激活持续时间 (秒)
        """
        if not self.initialized:
            logger.error("I/O 未初始化，无法激活筛除器")
            return False
        
        try:
            if self.mode == IOMode.GPIO:
                # 启动筛除器
                GPIO.output(self.rejector_control_pin, GPIO.HIGH)
                
                # 使用线程在指定时间后关闭
                threading.Timer(duration, 
                               lambda: GPIO.output(self.rejector_control_pin, GPIO.LOW)
                              ).start()
                
            elif self.mode == IOMode.MODBUS:
                # 假设使用线圈 1 控制筛除器
                self.modbus_client.write_bit(1, 1)
                
                # 使用线程在指定时间后关闭
                threading.Timer(duration, 
                               lambda: self.modbus_client.write_bit(1, 0)
                              ).start()
            
            logger.info(f"筛除器已激活 (持续 {duration} 秒)")
            return True
            
        except Exception as e:
            logger.error(f"激活筛除器失败: {e}")
            return False
    
    def check_position_sensor(self) -> bool:
        """
        检查位置传感器当前状态
        
        Returns:
            bool: 传感器是否被触发
        """
        if not self.initialized:
            logger.error("I/O 未初始化，无法读取位置传感器")
            return False
        
        try:
            if self.mode == IOMode.GPIO:
                # 读取 GPIO 状态
                return not GPIO.input(self.position_sensor_pin)  # 假设低电平为触发
            elif self.mode == IOMode.MODBUS:
                # 读取 Modbus 位状态
                return bool(self.modbus_client.read_bit(0))
            else:
                # 模拟模式，随机返回
                import random
                return random.random() < 0.1
                
        except Exception as e:
            logger.error(f"读取位置传感器失败: {e}")
            return False
    
    def close(self):
        """清理 I/O 资源"""
        try:
            # 停止监控线程
            self.running = False
            if self.position_thread and self.position_thread.is_alive():
                self.position_thread.join(timeout=1.0)
            
            # 停止传送带
            if self.is_conveyor_running:
                self.stop_conveyor()
            
            # 清理 GPIO 资源
            if self.mode == IOMode.GPIO and USE_JETSON_GPIO:
                GPIO.cleanup()
            
            # 关闭 Modbus 连接
            if self.mode == IOMode.MODBUS and self.modbus_client:
                self.modbus_client.serial.close()
            
            logger.info("I/O 控制器已关闭")
            
        except Exception as e:
            logger.error(f"关闭 I/O 资源时出错: {e}")
        finally:
            self.initialized = False

# 示例使用
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 检查是否在 Jetson 上运行
    on_jetson = os.path.exists('/proc/device-tree/model') and 'NVIDIA' in open('/proc/device-tree/model').read()
    
    if on_jetson:
        logger.info("在 Jetson 平台上运行")
        io_mode = IOMode.GPIO
    else:
        logger.info("非 Jetson 平台，使用模拟模式")
        io_mode = IOMode.SIMULATE
    
    # 创建 I/O 控制器
    io = IOController(mode=io_mode)
    
    # 注册位置传感器回调
    def position_callback():
        logger.info("位置传感器触发！")
        # 对于测试，触发时激活筛除器
        io.activate_rejector(0.5)
    
    io.register_position_callback(position_callback)
    
    try:
        # 启动传送带
        logger.info("启动传送带...")
        io.start_conveyor()
        
        # 运行一段时间
        logger.info("系统运行中，按 Ctrl+C 退出...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在退出...")
    finally:
        # 清理资源
        io.close()