# -*- coding: utf-8 -*-
"""
产品跟踪器 - 管理传送带上的产品信息并协调拍照和剔除操作
"""

import time
import logging
import threading
import queue
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from collections import deque

logger = logging.getLogger("DefectInspection.ProductTracker")

class ProductInfo:
    """保存单个产品的检测信息"""
    
    def __init__(self, product_id: int):
        self.product_id = product_id  # 产品唯一ID
        self.timestamp = datetime.now()  # 进入系统的时间戳
        self.image = None  # 产品图像
        self.detections = []  # 检测结果
        self.has_defect = False  # 是否有缺陷
        self.processing_status = "waiting"  # 处理状态: waiting, processing, finished, ejected
        self.cropped_results = []  # 裁剪结果
        self.upload_status = "waiting"  # 上传状态: waiting, uploaded, failed
    
    def __str__(self):
        return f"Product {self.product_id} (Status: {self.processing_status}, Defect: {self.has_defect})"


class ProductTracker:
    """
    产品跟踪器 - 管理传送带上多个产品的信息和状态
    
    功能:
    1. 跟踪拍照位置和剔除位置的产品
    2. 管理产品队列和状态更新
    3. 协调拍照、检测和剔除操作
    """
    
    def __init__(self, max_products: int = 100):
        # 最大跟踪产品数量
        self.max_products = max_products
        
        # 当前产品ID计数器
        self.current_id = 0
        
        # 传送带上的产品队列 (从拍照位置到剔除位置)
        self.product_queue = deque(maxlen=max_products)
        
        # 待处理产品队列 (等待检测)
        self.processing_queue = queue.Queue()
        
        # 剔除队列 (等待剔除)
        self.ejection_queue = deque(maxlen=max_products)
        
        # 已处理完成的产品历史
        self.product_history = {}
        
        # 状态变量
        self.running = True
        
        # 同步锁
        self.queue_lock = threading.Lock()
        
        # 回调函数
        self.on_product_detected = None  # 产品检测完成回调
        self.on_status_update = None  # 状态更新回调
        
        # 用于线程同步的事件
        self.new_product_event = threading.Event()
        self.new_ejection_event = threading.Event()
        
        # 统计信息
        self.stats = {
            "total_products": 0,
            "defect_products": 0,
            "ejected_products": 0,
            "processed_products": 0
        }
    
    def register_callbacks(self, on_product_detected: Callable = None, on_status_update: Callable = None):
        """注册回调函数"""
        self.on_product_detected = on_product_detected
        self.on_status_update = on_status_update
    
    def update_status(self, message: str, is_error: bool = False):
        """更新状态并调用回调函数"""
        if self.on_status_update:
            self.on_status_update(message, is_error)
        else:
            if is_error:
                logger.error(message)
            else:
                logger.info(message)
    
    def photo_sensor_triggered(self) -> int:
        """
        拍照位置传感器触发时调用
        
        Returns:
            int: 新产品的ID
        """
        with self.queue_lock:
            # 为新产品分配ID
            self.current_id += 1
            product_id = self.current_id
            
            # 创建新产品信息对象
            product = ProductInfo(product_id)
            
            # 添加到产品队列
            self.product_queue.append(product)
            
            # 更新统计信息
            self.stats["total_products"] += 1
            
            logger.info(f"拍照传感器检测到产品 (ID: {product_id})")
            self.update_status(f"拍照传感器检测到产品 (ID: {product_id})")
            
            return product_id
    
    def set_product_image(self, product_id: int, image) -> bool:
        """
        设置产品图像并添加到处理队列
        
        Args:
            product_id: 产品ID
            image: 产品图像
        
        Returns:
            bool: 操作是否成功
        """
        product = self.find_product_by_id(product_id)
        if not product:
            logger.error(f"无法找到产品 ID: {product_id}")
            return False
        
        # 设置图像并更新状态
        product.image = image
        product.processing_status = "processing"
        
        # 添加到处理队列
        self.processing_queue.put(product_id)
        
        # 触发新产品事件
        self.new_product_event.set()
        
        logger.debug(f"产品 {product_id} 图像已捕获，加入处理队列")
        return True
    
    def update_product_detection(self, product_id: int, detections: List, 
                                has_defect: bool, cropped_results: List = None) -> bool:
        """
        更新产品检测结果
        
        Args:
            product_id: 产品ID
            detections: 检测结果
            has_defect: 是否有缺陷
            cropped_results: 裁剪结果
        
        Returns:
            bool: 操作是否成功
        """
        product = self.find_product_by_id(product_id)
        if not product:
            logger.error(f"无法找到产品 ID: {product_id}")
            return False
        
        # 更新检测结果
        product.detections = detections
        product.has_defect = has_defect
        if cropped_results:
            product.cropped_results = cropped_results
        
        # 更新状态
        product.processing_status = "finished"
        
        # 更新统计信息
        self.stats["processed_products"] += 1
        if has_defect:
            self.stats["defect_products"] += 1
        
        # 如果有缺陷，添加到剔除队列
        if has_defect:
            self.ejection_queue.append(product_id)
            self.new_ejection_event.set()
            logger.info(f"产品 {product_id} 检测到瑕疵，已加入剔除队列")
        else:
            logger.info(f"产品 {product_id} 未检测到瑕疵")
        
        # 调用回调函数
        if self.on_product_detected:
            self.on_product_detected(product_id, has_defect, detections)
        
        return True
    
    def ejection_sensor_triggered(self) -> Optional[int]:
        """
        剔除位置传感器触发时调用
        
        Returns:
            Optional[int]: 需要剔除的产品ID，如果不需要剔除则返回None
        """
        with self.queue_lock:
            # 检查产品队列是否为空
            if not self.product_queue:
                logger.warning("剔除传感器触发，但产品队列为空")
                return None
            
            # 获取队列最前面的产品
            product = self.product_queue.popleft()
            product_id = product.product_id
            
            # 检查是否需要剔除
            if product_id in self.ejection_queue:
                # 从剔除队列中移除
                self.ejection_queue.remove(product_id)
                
                # 更新状态
                product.processing_status = "ejected"
                
                # 更新统计信息
                self.stats["ejected_products"] += 1
                
                # 添加到产品历史
                self.product_history[product_id] = product
                
                logger.info(f"产品 {product_id} 被标记为剔除")
                self.update_status(f"剔除瑕疵产品 (ID: {product_id})")
                
                return product_id
            else:
                # 添加到产品历史
                self.product_history[product_id] = product
                
                logger.debug(f"产品 {product_id} 通过检测，无需剔除")
                return None
    
    def find_product_by_id(self, product_id: int) -> Optional[ProductInfo]:
        """
        根据ID查找产品信息
        
        Args:
            product_id: 产品ID
        
        Returns:
            Optional[ProductInfo]: 产品信息对象
        """
        # 首先在当前队列中查找
        for product in self.product_queue:
            if product.product_id == product_id:
                return product
        
        # 然后在历史记录中查找
        if product_id in self.product_history:
            return self.product_history[product_id]
        
        return None
    
    def get_next_product_for_processing(self) -> Optional[int]:
        """
        从处理队列中获取下一个待处理产品ID
        
        Returns:
            Optional[int]: 待处理产品ID，如果队列为空则返回None
        """
        try:
            # 非阻塞获取
            return self.processing_queue.get(block=False)
        except queue.Empty:
            return None
    
    def wait_for_new_product(self, timeout: float = None) -> bool:
        """
        等待新产品加入处理队列
        
        Args:
            timeout: 超时时间(秒)
        
        Returns:
            bool: 是否有新产品
        """
        # 清除事件标志
        self.new_product_event.clear()
        # 等待事件
        return self.new_product_event.wait(timeout)
    
    def wait_for_ejection(self, timeout: float = None) -> bool:
        """
        等待有产品需要剔除
        
        Args:
            timeout: 超时时间(秒)
        
        Returns:
            bool: 是否有产品需要剔除
        """
        # 清除事件标志
        self.new_ejection_event.clear()
        # 等待事件
        return self.new_ejection_event.wait(timeout)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取当前统计信息"""
        return {
            "total_products": self.stats["total_products"],
            "defect_products": self.stats["defect_products"],
            "ejected_products": self.stats["ejected_products"],
            "processed_products": self.stats["processed_products"],
            "queued_products": len(self.product_queue),
            "processing_queue": self.processing_queue.qsize(),
            "ejection_queue": len(self.ejection_queue)
        }
    
    def shutdown(self):
        """关闭跟踪器"""
        self.running = False
        # 触发所有事件，使等待的线程可以退出
        self.new_product_event.set()
        self.new_ejection_event.set()