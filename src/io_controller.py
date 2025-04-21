#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单流水线 Jetson I/O控制模块

核心功能:
1. 位置传感器检测（基于中断）
2. 传送带控制（启动/停止）
3. 剔除执行器控制
"""

import os
import time
import logging
import threading
import subprocess
from typing import Callable, List
import Jetson.GPIO as GPIO

# 配置日志
logger = logging.getLogger("DefectInspection.IO")

class PipelineController:
    """单流水线控制器"""
    
    def __init__(self, 
                 position_sensor_pin: int, 
                 conveyor_pin: int, 
                 rejector_pin: int,
                 position_register: str,
                 conveyor_register: str,
                 rejector_register: str):
        """
        初始化单流水线控制器
        
        Args:
            position_sensor_pin: 位置传感器的GPIO引脚编号
            conveyor_pin: 传送带控制的GPIO引脚编号
            rejector_pin: 剔除执行器的GPIO引脚编号
            position_register: 位置传感器的寄存器配置
            conveyor_register: 传送带控制的寄存器配置
            rejector_register: 剔除执行器的寄存器配置
        """
        # 引脚配置
        self.position_sensor_pin = position_sensor_pin
        self.conveyor_pin = conveyor_pin
        self.rejector_pin = rejector_pin
        
        # 寄存器映射
        self.pin_register_map = {
            position_sensor_pin: position_register,
            conveyor_pin: conveyor_register,
            rejector_pin: rejector_register
        }
        
        # 状态变量
        self.initialized = False
        self.conveyor_running = False
        self.last_position_trigger = 0
        
        # 位置传感器回调
        self.position_callbacks = []
        
        # 初始化I/O
        self._initialize_io()
        
    def _initialize_register(self, pin):
        """
        初始化引脚对应的寄存器
        
        Args:
            pin: GPIO引脚编号(BOARD模式)
        """
        if pin in self.pin_register_map:
            cmd = f"sudo busybox devmem {self.pin_register_map[pin]}"
            try:
                logger.info(f"执行寄存器初始化: {cmd}")
                subprocess.run(cmd, shell=True, check=True)
                logger.debug(f"引脚 {pin} 寄存器初始化成功")
            except subprocess.CalledProcessError as e:
                logger.error(f"引脚 {pin} 寄存器初始化失败: {e}")
                return False
            return True
        else:
            logger.warning(f"引脚 {pin} 没有对应的寄存器配置")
            return False
        
    def _initialize_io(self):
        """初始化I/O系统"""
        try:
            # 设置GPIO为BOARD模式
            GPIO.setmode(GPIO.BOARD)
            
            # 初始化所有引脚的寄存器
            for pin in self.pin_register_map:
                self._initialize_register(pin)
            
            # 配置位置传感器为输入，显式设置上拉电阻
            GPIO.setup(self.position_sensor_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            # 配置传送带和执行器为输出，初始低电平
            GPIO.setup(self.conveyor_pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.rejector_pin, GPIO.OUT, initial=GPIO.LOW)
            
            # 读取初始状态以清除任何挂起的事件
            initial_state = GPIO.input(self.position_sensor_pin)
            logger.debug(f"位置传感器初始状态: {'HIGH' if initial_state else 'LOW'}")
            
            # 为位置传感器添加中断检测（下降沿触发）
            GPIO.add_event_detect(
                self.position_sensor_pin, 
                GPIO.FALLING, 
                callback=self._position_sensor_callback,
                bouncetime=300  # 增加防抖时间
            )
            
            logger.info("流水线GPIO初始化完成")
            
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
            
        # 重新读取引脚状态确认是真实触发
        pin_state = GPIO.input(self.position_sensor_pin)
        if pin_state == GPIO.HIGH:  # 如果是高电平，可能是误触发
            logger.debug("忽略可能的误触发")
            return
            
        # 防抖处理
        current_time = time.time()
        if current_time - self.last_position_trigger < 0.2:  # 增加到200ms
            logger.debug("忽略短时间内的重复触发")
            return
            
        self.last_position_trigger = current_time
        
        logger.info(f"位置传感器触发 (引脚 {channel})")
        
        # 执行所有注册的回调
        for callback in self.position_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"位置传感器回调执行错误: {e}")
    
    def register_position_callback(self, callback):
        """
        注册位置传感器回调函数
        
        Args:
            callback: 传感器触发时调用的函数
        """
        if callback not in self.position_callbacks:
            self.position_callbacks.append(callback)
            logger.debug(f"已注册位置传感器回调 (共 {len(self.position_callbacks)} 个)")
            return True
        return False
    
    def start_conveyor(self):
        """启动传送带"""
        if not self.initialized:
            logger.error("I/O未初始化，无法启动传送带")
            return False
            
        try:
            GPIO.output(self.conveyor_pin, GPIO.HIGH)
            
            self.conveyor_running = True
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
            GPIO.output(self.conveyor_pin, GPIO.LOW)
            
            self.conveyor_running = False
            logger.info("传送带已停止")
            return True
            
        except Exception as e:
            logger.error(f"停止传送带失败: {e}")
            return False
    
    def activate_rejector(self, duration=0.5):
        """
        激活剔除执行器
        
        Args:
            duration: 激活持续时间(秒)
        """
        if not self.initialized:
            logger.error("I/O未初始化，无法激活执行器")
            return False
            
        try:
            # 激活执行器
            GPIO.output(self.rejector_pin, GPIO.HIGH)
            
            # 使用定时器在指定时间后关闭
            threading.Timer(
                duration, 
                lambda: GPIO.output(self.rejector_pin, GPIO.LOW)
            ).start()
        
            logger.info(f"执行器已激活 (持续 {duration} 秒)")
            return True
            
        except Exception as e:
            logger.error(f"激活执行器失败: {e}")
            return False
    
    def close(self):
        """清理I/O资源"""
        try:
            # 停止传送带
            if self.conveyor_running:
                self.stop_conveyor()
            
            # 清理GPIO资源
            GPIO.cleanup([self.position_sensor_pin, self.conveyor_pin, self.rejector_pin])
            
            logger.info("流水线控制器已关闭")
            
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
    
    # 流水线1配置
    pipeline = PipelineController(
        position_sensor_pin=7,  # BOARD模式引脚7
        conveyor_pin=29,        # BOARD模式引脚29
        rejector_pin=31,        # BOARD模式引脚31
        position_register="0x02448030 w 0x040",
        conveyor_register="0x02430068 w 0x004",
        rejector_register="0x02430070 w 0x004"
    )
    
    # 注册位置传感器回调
    def position_callback():
        logger.info("检测到产品！")
        pipeline.stop_conveyor()
        # 模拟检测过程
        time.sleep(1)
        # 模拟剔除不良品
        pipeline.activate_rejector(3)
        # 重启传送带
        time.sleep(3)
        pipeline.start_conveyor()
    
    # 注册回调函数
    pipeline.register_position_callback(position_callback)
    
    try:
        # 启动传送带
        logger.info("启动流水线传送带...")
        pipeline.start_conveyor()
        
        # 运行一段时间
        logger.info("系统运行中，按Ctrl+C退出...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在退出...")
    finally:
        # 清理资源
        pipeline.close()