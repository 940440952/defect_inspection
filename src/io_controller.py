#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单流水线 Jetson I/O控制模块

核心功能:
1. 拍照位置传感器检测（基于中断）
2. 剔除位置传感器检测（基于中断）
3. 传送带控制（启动/停止）
4. 剔除执行器控制
"""

import os
import time
import logging
import threading
import subprocess
from typing import Callable, List, Optional, Dict
import Jetson.GPIO as GPIO

# 配置日志
logger = logging.getLogger("DefectInspection.IO")

class PipelineController:
    """单流水线控制器，支持拍照位置和剔除位置双传感器"""
    
    def __init__(self, 
                 photo_sensor_pin: int, 
                 conveyor_pin: int, 
                 rejector_pin: int,
                 ejection_sensor_pin: int,
                 photo_sensor_register: Optional[str] = None,
                 conveyor_register: Optional[str] = None,
                 rejector_register: Optional[str] = None,
                 ejection_sensor_register: Optional[str] = None):
        """
        初始化单流水线控制器
        
        Args:
            photo_sensor_pin: 拍照位置传感器的GPIO引脚编号
            conveyor_pin: 传送带控制的GPIO引脚编号
            rejector_pin: 剔除执行器的GPIO引脚编号
            ejection_sensor_pin: 剔除位置传感器的GPIO引脚编号
            photo_sensor_register: 拍照位置传感器的寄存器配置
            conveyor_register: 传送带控制的寄存器配置
            rejector_register: 剔除执行器的寄存器配置
            ejection_sensor_register: 剔除位置传感器的寄存器配置
        """
        # 引脚配置
        self.photo_sensor_pin = photo_sensor_pin
        self.conveyor_pin = conveyor_pin
        self.rejector_pin = rejector_pin
        self.ejection_sensor_pin = ejection_sensor_pin
        
        # 寄存器映射
        self.pin_register_map = {}
        if photo_sensor_register:
            self.pin_register_map[photo_sensor_pin] = photo_sensor_register
        if conveyor_register:
            self.pin_register_map[conveyor_pin] = conveyor_register
        if rejector_register:
            self.pin_register_map[rejector_pin] = rejector_register
        if ejection_sensor_register:
            self.pin_register_map[ejection_sensor_pin] = ejection_sensor_register
        
        # 状态变量
        self.initialized = False
        self.conveyor_running = False
        self.last_photo_trigger = 0
        self.last_ejection_trigger = 0
        
        # 传感器回调
        self.photo_sensor_callbacks = []  # 拍照位置传感器回调列表
        self.ejection_sensor_callbacks = []  # 剔除位置传感器回调列表
        
        # 初始化I/O
        self._initialize_io()
        
    def _initialize_register(self, pin):
        """
        初始化引脚对应的寄存器
        
        Args:
            pin: GPIO引脚编号(BOARD模式)
            
        Returns:
            bool: 初始化是否成功
        """
        if pin in self.pin_register_map:
            cmd = f"sudo busybox devmem {self.pin_register_map[pin]}"
            try:
                logger.info(f"执行寄存器初始化: {cmd}")
                subprocess.run(cmd, shell=True, check=True)
                logger.debug(f"引脚 {pin} 寄存器初始化成功")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"引脚 {pin} 寄存器初始化失败: {e}")
                return False
        else:
            logger.debug(f"引脚 {pin} 没有对应的寄存器配置，跳过寄存器初始化")
            return True  # 没有寄存器配置也视为成功
        
    def _initialize_io(self):
        """初始化I/O系统"""
        try:
            # 设置GPIO为BOARD模式
            GPIO.setmode(GPIO.BOARD)
            
            # 初始化所有引脚的寄存器
            for pin in self.pin_register_map:
                self._initialize_register(pin)
            
            # 配置位置传感器为输入，显式设置上拉电阻
            GPIO.setup(self.photo_sensor_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(self.ejection_sensor_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            # 配置传送带和执行器为输出，初始低电平
            GPIO.setup(self.conveyor_pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.rejector_pin, GPIO.OUT, initial=GPIO.LOW)
            
            # 读取初始状态以清除任何挂起的事件
            photo_initial_state = GPIO.input(self.photo_sensor_pin)
            ejection_initial_state = GPIO.input(self.ejection_sensor_pin)
            logger.debug(f"拍照传感器初始状态: {'HIGH' if photo_initial_state else 'LOW'}")
            logger.debug(f"剔除传感器初始状态: {'HIGH' if ejection_initial_state else 'LOW'}")
            
            # 为拍照位置传感器添加中断检测（下降沿触发）
            GPIO.add_event_detect(
                self.photo_sensor_pin, 
                GPIO.FALLING, 
                callback=self._photo_sensor_callback,
                bouncetime=300  # 防抖时间
            )
            
            # 为剔除位置传感器添加中断检测（下降沿触发）
            GPIO.add_event_detect(
                self.ejection_sensor_pin, 
                GPIO.FALLING, 
                callback=self._ejection_sensor_callback,
                bouncetime=300  # 防抖时间
            )
            
            logger.info("流水线GPIO初始化完成")
            
            # 标记初始化完成
            self.initialized = True
            
        except Exception as e:
            logger.error(f"I/O初始化失败: {e}")
            self.initialized = False    

    def _photo_sensor_callback(self, channel):
        """
        拍照传感器中断回调函数
        
        Args:
            channel: 触发中断的GPIO通道
        """
        if not self.initialized:
            return
            
        # 重新读取引脚状态确认是真实触发
        pin_state = GPIO.input(self.photo_sensor_pin)
        if pin_state == GPIO.HIGH:  # 如果是高电平，可能是误触发
            logger.debug("忽略拍照传感器可能的误触发")
            return
            
        # 防抖处理
        current_time = time.time()
        if current_time - self.last_photo_trigger < 0.2:  # 200ms防抖
            logger.debug("忽略拍照传感器短时间内的重复触发")
            return
            
        self.last_photo_trigger = current_time
        
        logger.info(f"拍照传感器触发 (引脚 {channel})")
        
        # 执行所有注册的回调
        for callback in self.photo_sensor_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"拍照传感器回调执行错误: {e}")
    
    def _ejection_sensor_callback(self, channel):
        """
        剔除传感器中断回调函数
        
        Args:
            channel: 触发中断的GPIO通道
        """
        if not self.initialized:
            return
            
        # 重新读取引脚状态确认是真实触发
        pin_state = GPIO.input(self.ejection_sensor_pin)
        if pin_state == GPIO.HIGH:  # 如果是高电平，可能是误触发
            logger.debug("忽略剔除传感器可能的误触发")
            return
            
        # 防抖处理
        current_time = time.time()
        if current_time - self.last_ejection_trigger < 0.2:  # 200ms防抖
            logger.debug("忽略剔除传感器短时间内的重复触发")
            return
            
        self.last_ejection_trigger = current_time
        
        logger.info(f"剔除传感器触发 (引脚 {channel})")
        
        # 执行所有注册的回调
        for callback in self.ejection_sensor_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"剔除传感器回调执行错误: {e}")
    
    def register_photo_callback(self, callback):
        """
        注册拍照传感器回调函数
        
        Args:
            callback: 传感器触发时调用的函数
            
        Returns:
            bool: 是否成功注册
        """
        if not self.initialized:
            logger.error("I/O未初始化，无法注册拍照传感器回调")
            return False
            
        if callback not in self.photo_sensor_callbacks:
            self.photo_sensor_callbacks.append(callback)
            logger.debug(f"已注册拍照传感器回调 (共 {len(self.photo_sensor_callbacks)} 个)")
            return True
        return False
    
    def register_ejection_callback(self, callback):
        """
        注册剔除传感器回调函数
        
        Args:
            callback: 传感器触发时调用的函数
            
        Returns:
            bool: 是否成功注册
        """
        if not self.initialized:
            logger.error("I/O未初始化，无法注册剔除传感器回调")
            return False
            
        if callback not in self.ejection_sensor_callbacks:
            self.ejection_sensor_callbacks.append(callback)
            logger.debug(f"已注册剔除传感器回调 (共 {len(self.ejection_sensor_callbacks)} 个)")
            return True
        return False
    
    def start_conveyor(self):
        """
        启动传送带
        
        Returns:
            bool: 操作是否成功
        """
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
        """
        停止传送带
        
        Returns:
            bool: 操作是否成功
        """
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
            
        Returns:
            bool: 操作是否成功
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
    
    def is_initialized(self):
        """
        检查I/O是否已初始化
        
        Returns:
            bool: I/O是否已初始化
        """
        return self.initialized
    
    def is_conveyor_running(self):
        """
        检查传送带是否运行中
        
        Returns:
            bool: 传送带是否运行中
        """
        return self.conveyor_running
    
    def get_status(self):
        """
        获取流水线状态信息
        
        Returns:
            Dict: 包含流水线状态的字典
        """
        return {
            "initialized": self.initialized,
            "conveyor_running": self.conveyor_running,
            "photo_callbacks": len(self.photo_sensor_callbacks),
            "ejection_callbacks": len(self.ejection_sensor_callbacks),
            "last_photo_trigger": self.last_photo_trigger,
            "last_ejection_trigger": self.last_ejection_trigger
        }
    
    def close(self):
        """
        清理I/O资源
        """
        if not self.initialized:
            return
            
        try:
            # 停止传送带
            if self.conveyor_running:
                self.stop_conveyor()
            
            # 清理GPIO事件检测
            GPIO.remove_event_detect(self.photo_sensor_pin)
            GPIO.remove_event_detect(self.ejection_sensor_pin)
            
            # 清理GPIO资源
            used_pins = [self.photo_sensor_pin, self.conveyor_pin, 
                         self.rejector_pin, self.ejection_sensor_pin]
            GPIO.cleanup(used_pins)
            
            logger.info("流水线控制器已关闭")
            
        except Exception as e:
            logger.error(f"关闭I/O资源时出错: {e}")
        finally:
            self.initialized = False
            self.conveyor_running = False
            self.photo_sensor_callbacks = []
            self.ejection_sensor_callbacks = []


# 测试代码
# if __name__ == "__main__":
#     # 配置日志
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )
    
#     # 导入随机模块（移到顶部，避免在回调函数内部导入）
#     import random
    
#     # 流水线控制器配置
#     pipeline = PipelineController(
#         photo_sensor_pin=7,      # 拍照传感器引脚
#         conveyor_pin=29,         # 传送带控制引脚
#         rejector_pin=31,         # 剔除执行器引脚
#         ejection_sensor_pin=32,  # 剔除传感器引脚
#         photo_sensor_register="0x02448030 w 0x040",  # 拍照传感器寄存器
#         conveyor_register="0x02430068 w 0x004",      # 传送带控制寄存器
#         rejector_register="0x02430070 w 0x004",      # 剔除执行器寄存器
#         ejection_sensor_register="0x02434080 w 0x040"  # 剔除传感器寄存器
#     )
    
#     logger.info("流水线控制器配置完成")
    
    # 测试传送带和剔除装置
    # pipeline.start_conveyor()
    # time.sleep(10)
    
    # time.sleep(3)

    # for i in range(3):
    #     logger.info(f"激活剔除装置 ({i+1}/3)...")
    #     result = pipeline.activate_rejector(0.5)
    #     logger.info(f"剔除装置激活结果: {'成功' if result else '失败'}")
    #     time.sleep(1.5)

    # 拍照传感器引脚测试
    # 测试计数
    # trigger_count = 0
    
    # # 定义回调函数
    # def sensor_callback():
    #     global trigger_count
    #     trigger_count += 1
    #     logger.info(f"传感器触发次数: {trigger_count}")
    
    # # 注册回调
    # pipeline.register_ejection_callback(sensor_callback)
    
    # # 测试持续时间(秒)
    # test_duration = 60
    
    # logger.info(f"等待传感器触发，测试将持续 {test_duration} 秒...")
    # logger.info("请手动触发传感器或放置物体通过传感器位置")
    
    # # 记录开始时间
    # start_time = time.time()
    
    # try:
    #     # 测试循环
    #     while time.time() - start_time < test_duration:
    #         # 每秒打印一次状态
    #         if int(time.time() - start_time) % 2 == 0:
    #             pin_state = GPIO.input(pipeline.photo_sensor_pin)
    #             logger.debug(f"传感器当前状态: {'HIGH' if pin_state else 'LOW'}")
    #         time.sleep(0.1)
            
    #     # 测试结束
    #     logger.info(f"测试完成，传感器共触发 {trigger_count} 次")
        
    # except Exception as e:
    #     logger.error(f"测试过程中发生错误: {e}")