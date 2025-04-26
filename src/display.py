#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
瑕疵检测显示界面模块

使用Tkinter实现的检测结果显示界面，包括：
- 左侧：摄像头图像和检测结果显示
- 右侧：检测状态和统计信息显示
"""

import os
import io
import time
import logging
import threading
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, Frame, Label, Button, Text, Scrollbar, Canvas
from PIL import Image, ImageTk
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import queue

# 设置日志
logger = logging.getLogger("DefectInspection.Display")

class DefectDisplay:
    """瑕疵检测显示界面类"""
    
    def __init__(self, window_title="瑕疵检测系统"):
        """
        初始化显示界面
        
        Args:
            window_title: 窗口标题
        """
        self.window_title = window_title
        self.root = None
        self.running = False
        self.latest_image = None
        self.photo_image = None  # 保存Tkinter PhotoImage对象
        self.latest_results = []
        self._error_status = False  # 新增：错误状态标志
        self.stats = {
            "total_inspections": 0,
            "defect_count": 0,
            "start_time": time.time(),
            "last_detection_time": 0,
            "defect_types": {
                "defect": 0  # 简化为只有一种缺陷类型
            }
        }
        
        # 系统状态
        self.system_status = "准备就绪"
        self.status_var = None  # Tkinter StringVar for status
        
        # 用于保存界面元素的引用
        self.ui_elements = {}
        
        # 进度条变量
        self.progress_vars = {}
        
        # 创建一个UI更新队列
        self.ui_update_queue = queue.Queue()
        
        # 创建窗口
        self._create_window()
    
    def _create_window(self):
        """创建Tkinter窗口"""
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title(self.window_title)
        self.root.geometry("1200x700")
        self.root.configure(bg="#333333")
        
        # 配置网格权重，使左右两列可以适当调整大小
        self.root.grid_columnconfigure(0, weight=3)  # 左侧图像区域
        self.root.grid_columnconfigure(1, weight=1)  # 右侧信息区域
        self.root.grid_rowconfigure(0, weight=1)
        
        # === 左侧图像显示区域 ===
        left_frame = Frame(self.root, bg="#222222", padx=10, pady=10)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # 图像标题
        Label(left_frame, text="检测图像", font=("微软雅黑", 14), bg="#222222", fg="white").pack(anchor="w", pady=(0, 10))
        
        # 图像显示区域 (Canvas)
        self.canvas = Canvas(left_frame, bg="black", width=640, height=480, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # 状态显示
        status_frame = Frame(left_frame, bg="#222222", pady=5)
        status_frame.pack(fill=tk.X, expand=False)
        
        Label(status_frame, text="检测状态:", font=("微软雅黑", 10), bg="#222222", fg="white").pack(side=tk.LEFT, padx=(0, 5))
        
        self.status_var = tk.StringVar(value="准备就绪")
        self.ui_elements["-STATUS-"] = Label(status_frame, textvariable=self.status_var, 
                                            font=("微软雅黑", 10, "bold"), bg="#222222", fg="white",
                                            anchor="w", width=50)
        self.ui_elements["-STATUS-"].pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # === 右侧信息显示区域 ===
        right_frame = Frame(self.root, bg="#222222", padx=10, pady=10)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # 标题
        Label(right_frame, text="系统信息", font=("微软雅黑", 14), bg="#222222", fg="white").pack(anchor="w", pady=(0, 10))
        
        # 1. 运行统计框架
        stats_frame = Frame(right_frame, bg="#333333", padx=8, pady=8, relief=tk.GROOVE, bd=2)
        stats_frame.pack(fill=tk.X, expand=True, pady=(0, 10))
        
        Label(stats_frame, text="运行统计", font=("微软雅黑", 12, "bold"), bg="#333333", fg="white").pack(anchor="w", pady=(0, 5))
        
        # 运行统计内容
        stats_content = Frame(stats_frame, bg="#333333")
        stats_content.pack(fill=tk.X, expand=True)
        
        # 开始时间
        Label(stats_content, text="开始时间:", font=("微软雅黑", 10), bg="#333333", fg="white").grid(row=0, column=0, sticky="w", pady=2)
        self.ui_elements["-START-TIME-"] = Label(stats_content, text="", font=("微软雅黑", 10), bg="#333333", fg="white")
        self.ui_elements["-START-TIME-"].grid(row=0, column=1, sticky="w", padx=(5, 0), pady=2)
        
        # 运行时长
        Label(stats_content, text="运行时长:", font=("微软雅黑", 10), bg="#333333", fg="white").grid(row=1, column=0, sticky="w", pady=2)
        self.ui_elements["-RUN-TIME-"] = Label(stats_content, text="00:00:00", font=("微软雅黑", 10), bg="#333333", fg="white")
        self.ui_elements["-RUN-TIME-"].grid(row=1, column=1, sticky="w", padx=(5, 0), pady=2)
        
        # 总检测数
        Label(stats_content, text="总检测数:", font=("微软雅黑", 10), bg="#333333", fg="white").grid(row=2, column=0, sticky="w", pady=2)
        self.ui_elements["-TOTAL-INSPECTIONS-"] = Label(stats_content, text="0", font=("微软雅黑", 10), bg="#333333", fg="white")
        self.ui_elements["-TOTAL-INSPECTIONS-"].grid(row=2, column=1, sticky="w", padx=(5, 0), pady=2)
        
        # 发现缺陷
        Label(stats_content, text="发现缺陷:", font=("微软雅黑", 10), bg="#333333", fg="white").grid(row=3, column=0, sticky="w", pady=2)
        self.ui_elements["-DEFECT-COUNT-"] = Label(stats_content, text="0", font=("微软雅黑", 10, "bold"), bg="#333333", fg="red")
        self.ui_elements["-DEFECT-COUNT-"].grid(row=3, column=1, sticky="w", padx=(5, 0), pady=2)
        
        # 缺陷率
        Label(stats_content, text="缺陷率:", font=("微软雅黑", 10), bg="#333333", fg="white").grid(row=4, column=0, sticky="w", pady=2)
        self.ui_elements["-DEFECT-RATE-"] = Label(stats_content, text="0.0%", font=("微软雅黑", 10), bg="#333333", fg="white")
        self.ui_elements["-DEFECT-RATE-"].grid(row=4, column=1, sticky="w", padx=(5, 0), pady=2)
        
        # 2. 缺陷类型统计框架 - 简化为只显示一种缺陷
        defect_frame = Frame(right_frame, bg="#333333", padx=8, pady=8, relief=tk.GROOVE, bd=2)
        defect_frame.pack(fill=tk.X, expand=True, pady=(0, 10))
        
        Label(defect_frame, text="缺陷统计", font=("微软雅黑", 12, "bold"), bg="#333333", fg="white").pack(anchor="w", pady=(0, 5))
        
        # 缺陷类型统计内容
        defect_content = Frame(defect_frame, bg="#333333")
        defect_content.pack(fill=tk.X, expand=True)
        
        # 只添加一种缺陷类型
        defect_id = "defect"
        defect_name = "瑕疵"
        
        # 创建缺陷类型标签
        Label(defect_content, text=f"{defect_name}:", font=("微软雅黑", 10), bg="#333333", fg="white").grid(row=0, column=0, sticky="w", pady=4)
        
        # 创建计数标签
        self.ui_elements[f"-{defect_id.upper()}-COUNT-"] = Label(defect_content, text="0", font=("微软雅黑", 10), bg="#333333", fg="white", width=5)
        self.ui_elements[f"-{defect_id.upper()}-COUNT-"].grid(row=0, column=1, sticky="w", padx=(5, 10), pady=4)
        
        # 创建进度条
        self.progress_vars[defect_id] = tk.DoubleVar(value=0)
        progress = ttk.Progressbar(defect_content, variable=self.progress_vars[defect_id], length=150, mode="determinate", maximum=100)
        progress.grid(row=0, column=2, sticky="w", pady=4)
        self.ui_elements[f"-{defect_id.upper()}-PROGRESS-"] = progress
        
        # 3. 最近检测框架
        recent_frame = Frame(right_frame, bg="#333333", padx=8, pady=8, relief=tk.GROOVE, bd=2)
        recent_frame.pack(fill=tk.X, expand=True, pady=(0, 10))
        
        Label(recent_frame, text="最近检测", font=("微软雅黑", 12, "bold"), bg="#333333", fg="white").pack(anchor="w", pady=(0, 5))
        
        # 最近检测内容
        recent_content = Frame(recent_frame, bg="#333333")
        recent_content.pack(fill=tk.X, expand=True)
        
        # 时间
        Label(recent_content, text="时间:", font=("微软雅黑", 10), bg="#333333", fg="white").grid(row=0, column=0, sticky="w", pady=2)
        self.ui_elements["-LAST-DETECTION-TIME-"] = Label(recent_content, text="--", font=("微软雅黑", 10), bg="#333333", fg="white")
        self.ui_elements["-LAST-DETECTION-TIME-"].grid(row=0, column=1, sticky="w", padx=(5, 0), pady=2)
        
        # 结果
        Label(recent_content, text="结果:", font=("微软雅黑", 10), bg="#333333", fg="white").grid(row=1, column=0, sticky="w", pady=2)
        self.ui_elements["-LAST-DETECTION-RESULT-"] = Label(recent_content, text="--", font=("微软雅黑", 10), bg="#333333", fg="white")
        self.ui_elements["-LAST-DETECTION-RESULT-"].grid(row=1, column=1, sticky="w", padx=(5, 0), pady=2)
        
        # 详情
        Label(recent_content, text="详情:", font=("微软雅黑", 10), bg="#333333", fg="white").grid(row=2, column=0, sticky="nw", pady=2)
        
        # 文本框和滚动条
        text_frame = Frame(recent_content, bg="#444444")
        text_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(2, 5))
        
        text_scroll = Scrollbar(text_frame)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.ui_elements["-DETECTION-DETAILS-"] = Text(text_frame, height=5, width=40, 
                                                      font=("微软雅黑", 9), bg="#444444", fg="white",
                                                      yscrollcommand=text_scroll.set)
        self.ui_elements["-DETECTION-DETAILS-"].pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scroll.config(command=self.ui_elements["-DETECTION-DETAILS-"].yview)
        
        self.ui_elements["-DETECTION-DETAILS-"].insert(tk.END, "暂无检测记录")
        self.ui_elements["-DETECTION-DETAILS-"].config(state=tk.DISABLED)
        
        # 4. 控制按钮
        button_frame = Frame(right_frame, bg="#222222", pady=10)
        button_frame.pack(fill=tk.X, expand=False)
        
        # 开始按钮
        self.ui_elements["-START-"] = Button(button_frame, text="开始检测", font=("微软雅黑", 10),
                                            bg="#007bff", fg="white", relief=tk.RAISED,
                                            command=self.start)
        self.ui_elements["-START-"].pack(side=tk.LEFT, padx=(0, 10))
        
        # 停止按钮
        self.ui_elements["-STOP-"] = Button(button_frame, text="停止检测", font=("微软雅黑", 10),
                                           bg="#dc3545", fg="white", relief=tk.RAISED,
                                           command=self.stop, state=tk.DISABLED)
        self.ui_elements["-STOP-"].pack(side=tk.LEFT, padx=(0, 10))
        
        # 退出按钮
        self.ui_elements["-EXIT-"] = Button(button_frame, text="退出", font=("微软雅黑", 10),
                                           bg="#6c757d", fg="white", relief=tk.RAISED,
                                           command=self.close)
        self.ui_elements["-EXIT-"].pack(side=tk.LEFT)
        
        # 设置开始时间
        self.ui_elements["-START-TIME-"].config(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # 配置关闭窗口事件
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        
        # 启动UI更新处理
        self._start_ui_update_handler()
    
    def _start_ui_update_handler(self):
        """启动处理UI更新队列的函数"""
        def process_ui_updates():
            try:
                while not self.ui_update_queue.empty():
                    update_func, args, kwargs = self.ui_update_queue.get_nowait()
                    try:
                        update_func(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"执行UI更新函数时出错: {e}")
                    finally:
                        self.ui_update_queue.task_done()
            except Exception as e:
                logger.error(f"处理UI更新队列时出错: {e}")
            finally:
                # 安排下一次处理
                if self.root and self.running:
                    self.root.after(50, process_ui_updates)
        
        # 启动第一次处理
        if self.root:
            self.root.after(50, process_ui_updates)
    
    def _queue_ui_update(self, update_func, *args, **kwargs):
        """将UI更新函数添加到队列"""
        self.ui_update_queue.put((update_func, args, kwargs))
    
    def start(self):
        """开始检测"""
        self.running = True
        self._queue_ui_update(self._update_buttons_state)
        logger.info("检测流程启动")
    
    def _update_buttons_state(self):
        """更新按钮状态 - 内部方法，只能在主线程调用"""
        if self.running:
            self.ui_elements["-START-"].config(state=tk.DISABLED)
            self.ui_elements["-STOP-"].config(state=tk.NORMAL)
        else:
            self.ui_elements["-START-"].config(state=tk.NORMAL)
            self.ui_elements["-STOP-"].config(state=tk.DISABLED)
    
    def stop(self):
        """停止检测"""
        self.running = False
        self._queue_ui_update(self._update_buttons_state)
        self._queue_ui_update(self._set_status_internal, "检测已停止", False)
        logger.info("检测流程停止")
    
    def set_status(self, status: str, is_error: bool = False):
        """
        设置系统状态 - 线程安全版本
        
        Args:
            status: 状态描述
            is_error: 是否为错误状态
        """
        self._queue_ui_update(self._set_status_internal, status, is_error)
    
    def _set_status_internal(self, status: str, is_error: bool = False):
        """内部使用，直接设置状态（在主线程中调用）"""
        self.system_status = status
        
        if self.status_var:
            self.status_var.set(status)
            if is_error:
                self.ui_elements["-STATUS-"].config(fg="red")
            else:
                self.ui_elements["-STATUS-"].config(fg="white")
    
    def update_image(self, image: np.ndarray, detection_results: List = None):
        """
        更新图像和检测结果 - 线程安全版本
        
        Args:
            image: 要显示的图像(OpenCV格式，BGR)，应该已经包含检测结果的绘制
            detection_results: 检测结果列表，每项为[x1, y1, x2, y2, conf, class_id]
        """
        if image is None:
            return False
            
        # 保存当前图像和检测结果，然后在主线程中处理
        self.latest_image = image.copy() if hasattr(image, 'copy') else image
        self.latest_results = detection_results
        
        # 更新统计信息
        self.stats["total_inspections"] += 1
        if detection_results and len(detection_results) > 0:
            self.stats["defect_count"] += 1
            self.stats["defect_types"]["defect"] += len(detection_results)
        self.stats["last_detection_time"] = time.time()
        
        # 将实际的UI更新放入队列
        self._queue_ui_update(self._update_image_internal)
        self._queue_ui_update(self._update_info)
        return True
    
    def _update_image_internal(self):
        """内部使用，实际执行图像更新（在主线程中调用）"""
        if not self.root or self.latest_image is None:
            return
            
        try:
            # 转换为PIL图像，然后转为Tkinter PhotoImage
            img_rgb = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # 调整图像大小以适应画布
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:  # 确保画布有有效尺寸
                # 计算图像缩放比例，保持纵横比
                img_width, img_height = pil_img.size
                scale = min(canvas_width/img_width, canvas_height/img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                
                # 缩放图像
                if scale != 1:
                    pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
            
            # 转换为Tkinter可用的格式
            self.photo_image = ImageTk.PhotoImage(pil_img)
            
            # 清除画布并显示新图像
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width // 2, canvas_height // 2,  # 居中显示
                image=self.photo_image, anchor=tk.CENTER
            )
            
            # 更新检测结果详情
            self._update_detection_details()
            
        except Exception as e:
            logger.error(f"更新图像时出错: {e}")
    
    def _update_detection_details(self):
        """更新检测结果详情（在主线程中调用）"""
        if not self.latest_results:
            return
            
        # 启用编辑
        self.ui_elements["-DETECTION-DETAILS-"].config(state=tk.NORMAL)
        self.ui_elements["-DETECTION-DETAILS-"].delete(1.0, tk.END)
        
        # 检测时间
        detection_time = datetime.fromtimestamp(self.stats["last_detection_time"])
        time_str = detection_time.strftime("%H:%M:%S")
        self.ui_elements["-LAST-DETECTION-TIME-"].config(text=time_str)
        
        # 检测结果
        has_defect = len(self.latest_results) > 0
        if has_defect:
            result_text = f"发现{len(self.latest_results)}个瑕疵"
            self.ui_elements["-LAST-DETECTION-RESULT-"].config(text=result_text, fg="red")
        else:
            result_text = "合格产品"
            self.ui_elements["-LAST-DETECTION-RESULT-"].config(text=result_text, fg="green")
        
        # 详情文本
        if has_defect:
            self.ui_elements["-DETECTION-DETAILS-"].insert(tk.END, f"检测时间: {time_str}\n")
            self.ui_elements["-DETECTION-DETAILS-"].insert(tk.END, f"检测结果: {result_text}\n\n")
            
            # 添加每个缺陷的详细信息
            for i, det in enumerate(self.latest_results):
                if len(det) >= 6:  # x1,y1,x2,y2,conf,class
                    conf = det[4]
                    cls_id = int(det[5])
                    cls_name = "瑕疵" if cls_id == 0 else f"类型{cls_id}"
                    self.ui_elements["-DETECTION-DETAILS-"].insert(
                        tk.END, f"[{i+1}] {cls_name}: 置信度 {conf:.2f}\n"
                    )
        else:
            self.ui_elements["-DETECTION-DETAILS-"].insert(tk.END, f"检测时间: {time_str}\n")
            self.ui_elements["-DETECTION-DETAILS-"].insert(tk.END, "检测结果: 未发现瑕疵\n")
        
        # 禁用编辑
        self.ui_elements["-DETECTION-DETAILS-"].config(state=tk.DISABLED)
    
    def _update_info(self):
        """更新界面上的统计信息（在主线程中调用）"""
        # 更新运行时长
        runtime = time.time() - self.stats["start_time"]
        hours, remainder = divmod(int(runtime), 3600)
        minutes, seconds = divmod(remainder, 60)
        self.ui_elements["-RUN-TIME-"].config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # 更新总检测数
        self.ui_elements["-TOTAL-INSPECTIONS-"].config(text=str(self.stats["total_inspections"]))
        
        # 更新缺陷数
        self.ui_elements["-DEFECT-COUNT-"].config(text=str(self.stats["defect_count"]))
        
        # 更新缺陷率
        if self.stats["total_inspections"] > 0:
            defect_rate = self.stats["defect_count"] / self.stats["total_inspections"] * 100
            self.ui_elements["-DEFECT-RATE-"].config(text=f"{defect_rate:.1f}%")
        
        # 更新缺陷类型统计
        # 这里只处理一种缺陷类型
        defect_count = self.stats["defect_types"].get("defect", 0)
        self.ui_elements["-DEFECT-COUNT-"].config(text=str(defect_count))
        
        # 更新进度条
        if self.stats["total_inspections"] > 0:
            defect_ratio = min(100, defect_count / max(1, self.stats["total_inspections"]) * 100)
            self.progress_vars["defect"].set(defect_ratio)
    
    def run(self, update_interval: float = 0.1):
        """
        运行主循环
        
        Args:
            update_interval: 界面更新间隔(秒)
        """
        if not self.root:
            logger.error("窗口未初始化")
            return
            
        try:
            # 定期更新函数
            def update():
                if self.running:
                    self._queue_ui_update(self._update_info)
                    self.root.after(int(update_interval * 1000), update)
            
            # 启动定期更新
            update()
            
            # 主循环
            self.root.mainloop()
                
        except Exception as e:
            logger.error(f"显示界面运行出错: {e}")
        finally:
            # 不要在这里销毁窗口，让main.py中的cleanup_system来处理
            logger.debug("显示界面主循环结束")
    
    def close(self):
        """关闭窗口"""
        # 确保在主线程中执行
        if self.root and threading.current_thread() is threading.main_thread():
            self.running = False
            self.root.destroy()
            self.root = None
        else:
            # 如果不在主线程，将关闭请求加入队列
            try:
                self._queue_ui_update(self.close)
            except:
                # 如果队列也无法访问，使用其他方法通知主线程
                logger.warning("在非主线程中调用close()，无法使用队列")
                self.running = False