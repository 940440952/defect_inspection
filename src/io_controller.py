#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双流水线 Jetson I/O控制模块

核心功能:
1. 位置传感器检测（基于中断）
2. 传送带控制（启动/停止）
3. 剔除执行器控制
4. 支持双流水线独立控制
"""

import os
import time
import logging
import threading
import subprocess
from typing import Callable, Dict, List
import Jetson.GPIO as GPIO

# 配置日志
logger = logging.getLogger("DefectInspection.IO")

    
# 定义引脚配置和寄存器映射
PIN_CONFIG = {
    # 流水线1配置
    1: {
        'position_sensor': 7,     # BOARD模式引脚7
        'conveyor_control': 29,   # BOARD模式引脚29
        'rejector_control': 31,   # BOARD模式引脚31
    },
    # 流水线2配置
    2: {
        'position_sensor': 32,    # BOARD模式引脚32
        'conveyor_control': 33,   # BOARD模式引脚33
        'rejector_control': 35,   # BOARD模式引脚35
    }
}

# 定义引脚寄存器映射
PIN_REGISTER_MAP = {
    # 输出引脚
    29: "0x02430068 w 0x004",  # 流水线1传送带控制
    31: "0x02430070 w 0x004",  # 流水线1执行器控制
    33: "0x02434040 w 0x004",  # 流水线2传送带控制
    35: "0x024340a0 w 0x004",  # 流水线2执行器控制
    
    # 输入引脚
    7: "0x02448030 w 0x040",   # 流水线1位置传感器
    32: "0x02434080 w 0x040",  # 流水线2位置传感器
}

class IOController:
    """双流水线 I/O控制器"""
    
    def __init__(self):
        """初始化双流水线I/O控制器"""
        # 状态变量
        self.initialized = False
        self.conveyor_running = {1: False, 2: False}
        self.last_position_trigger = {1: 0, 2: 0}
        
        # 位置传感器回调
        self.position_callbacks = {1: [], 2: []}
        
        # 初始化I/O
        self._initialize_io()
        
    def _initialize_register(self, pin):
        """
        初始化引脚对应的寄存器
        
        Args:
            pin: GPIO引脚编号(BOARD模式)
        """
        if pin in PIN_REGISTER_MAP:
            cmd = f"sudo busybox devmem {PIN_REGISTER_MAP[pin]}"
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
            for line_id in [1, 2]:
                for pin_type, pin in PIN_CONFIG[line_id].items():
                    self._initialize_register(pin)
            
            # 配置GPIO引脚
            for line_id in [1, 2]:
                config = PIN_CONFIG[line_id]
                
                # 配置位置传感器为输入，带内部上拉
                GPIO.setup(config['position_sensor'], GPIO.IN, pull_up_down=GPIO.PUD_UP)
                
                # 配置传送带和执行器为输出，初始低电平
                GPIO.setup(config['conveyor_control'], GPIO.OUT, initial=GPIO.LOW)
                GPIO.setup(config['rejector_control'], GPIO.OUT, initial=GPIO.LOW)
                
                # 为位置传感器添加中断检测（下降沿触发）
                GPIO.add_event_detect(
                    config['position_sensor'], 
                    GPIO.FALLING, 
                    callback=lambda channel, line=line_id: self._position_sensor_callback(channel, line),
                    bouncetime=200
                )
                
                logger.info(f"流水线 {line_id} GPIO初始化完成")
            
            # 标记初始化完成
            self.initialized = True
            
        except Exception as e:
            logger.error(f"I/O初始化失败: {e}")
            self.initialized = False
    
    def _position_sensor_callback(self, channel, line_id):
        """
        位置传感器中断回调函数
        
        Args:
            channel: 触发中断的GPIO通道
            line_id: 流水线ID (1或2)
        """
        if not self.initialized:
            return
            
        # 防抖处理
        current_time = time.time()
        if current_time - self.last_position_trigger[line_id] < 0.1:
            return
            
        self.last_position_trigger[line_id] = current_time
        
        logger.debug(f"流水线 {line_id} 位置传感器触发 (引脚 {channel})")
        
        # 执行所有注册的回调
        for callback in self.position_callbacks[line_id]:
            try:
                callback(line_id)
            except Exception as e:
                logger.error(f"位置传感器回调执行错误: {e}")
    
    def register_position_callback(self, line_id, callback):
        """
        注册位置传感器回调函数
        
        Args:
            line_id: 流水线ID (1或2)
            callback: 传感器触发时调用的函数，将接收line_id参数
        """
        if line_id not in [1, 2]:
            logger.error(f"无效的流水线ID: {line_id}")
            return False
            
        if callback not in self.position_callbacks[line_id]:
            self.position_callbacks[line_id].append(callback)
            logger.debug(f"已为流水线 {line_id} 注册位置传感器回调 (共 {len(self.position_callbacks[line_id])} 个)")
            return True
        return False
    
    def start_conveyor(self, line_id):
        """
        启动传送带
        
        Args:
            line_id: 流水线ID (1或2)
        """
        if not self.initialized:
            logger.error("I/O未初始化，无法启动传送带")
            return False
            
        if line_id not in [1, 2]:
            logger.error(f"无效的流水线ID: {line_id}")
            return False
        
        try:
            conveyor_pin = PIN_CONFIG[line_id]['conveyor_control']
            GPIO.output(conveyor_pin, GPIO.HIGH)
            
            self.conveyor_running[line_id] = True
            logger.info(f"流水线 {line_id} 传送带已启动")
            return True
            
        except Exception as e:
            logger.error(f"启动流水线 {line_id} 传送带失败: {e}")
            return False
    
    def stop_conveyor(self, line_id):
        """
        停止传送带
        
        Args:
            line_id: 流水线ID (1或2)
        """
        if not self.initialized:
            logger.error("I/O未初始化，无法停止传送带")
            return False
            
        if line_id not in [1, 2]:
            logger.error(f"无效的流水线ID: {line_id}")
            return False
        
        try:
            conveyor_pin = PIN_CONFIG[line_id]['conveyor_control']
            GPIO.output(conveyor_pin, GPIO.LOW)
            
            self.conveyor_running[line_id] = False
            logger.info(f"流水线 {line_id} 传送带已停止")
            return True
            
        except Exception as e:
            logger.error(f"停止流水线 {line_id} 传送带失败: {e}")
            return False
    
    def activate_rejector(self, line_id, duration=0.5):
        """
        激活剔除执行器
        
        Args:
            line_id: 流水线ID (1或2)
            duration: 激活持续时间(秒)
        """
        if not self.initialized:
            logger.error("I/O未初始化，无法激活执行器")
            return False
            
        if line_id not in [1, 2]:
            logger.error(f"无效的流水线ID: {line_id}")
            return False
        
        try:
            rejector_pin = PIN_CONFIG[line_id]['rejector_control']
            
            # 激活执行器
            GPIO.output(rejector_pin, GPIO.HIGH)
            
            # 使用定时器在指定时间后关闭
            threading.Timer(
                duration, 
                lambda: GPIO.output(rejector_pin, GPIO.LOW)
            ).start()
        
            logger.info(f"流水线 {line_id} 执行器已激活 (持续 {duration} 秒)")
            return True
            
        except Exception as e:
            logger.error(f"激活流水线 {line_id} 执行器失败: {e}")
            return False
    
    def close(self):
        """清理I/O资源"""
        try:
            # 停止所有传送带
            for line_id in [1, 2]:
                if self.conveyor_running[line_id]:
                    self.stop_conveyor(line_id)
            
            # 清理GPIO资源
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
    
    # 创建I/O控制器
    io = IOController()
    
    # 注册位置传感器回调
    def position_callback(line_id):
        logger.info(f"流水线 {line_id} 检测到产品！")
        io.stop_conveyor(line_id)
        # 模拟检测过程
        time.sleep(1)
        # 模拟剔除不良品
        io.activate_rejector(line_id, 0.5)
        # 重启传送带
        time.sleep(0.5)
        io.start_conveyor(line_id)
    
    # 为两条流水线注册相同的回调函数
    io.register_position_callback(1, position_callback)
    io.register_position_callback(2, position_callback)
    
    try:
        # 启动两条流水线的传送带
        logger.info("启动所有流水线传送带...")
        io.start_conveyor(1)
        io.start_conveyor(2)
        
        # 运行一段时间
        logger.info("系统运行中，按Ctrl+C退出...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在退出...")
    finally:
        # 清理资源
        io.close()