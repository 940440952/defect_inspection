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
    
    def update_image(self, image: np.ndarray, detection_results: List = None):
        """
        更新图像和检测结果
        
        Args:
            image: 要显示的图像(OpenCV格式，BGR)，应该已经包含检测结果的绘制
            detection_results: 检测结果列表，每项为[x1, y1, x2, y2, conf, class_id]
        """
        if image is None or self.root is None:
            return False
            
        # 保存最新图像
        self.latest_image = image.copy()
        
        # 处理检测结果
        if detection_results is not None:
            self.latest_results = detection_results
            self.stats["last_detection_time"] = time.time()
            # 更新统计信息
            self.stats["total_inspections"] += 1
            
            # 检查是否有缺陷
            if len(detection_results) > 0:
                self.stats["defect_count"] += 1
                # 所有缺陷计入同一类别
                self.stats["defect_types"]["defect"] += len(detection_results)
        else:
            # 没有检测结果但仍计为一次检测
            self.stats["total_inspections"] += 1
            
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
        
        # 更新界面信息
        self._update_info()
        return True
    
    def _update_info(self):
        """更新界面信息"""
        if not self.root:
            return
            
        # 计算运行时间
        run_time = time.time() - self.stats["start_time"]
        hours, remainder = divmod(int(run_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        run_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # 计算缺陷率
        defect_rate = 0.0
        if self.stats["total_inspections"] > 0:
            defect_rate = (self.stats["defect_count"] / self.stats["total_inspections"]) * 100
            
        # 更新最近检测时间
        if self.stats["last_detection_time"] > 0:
            time_diff = time.time() - self.stats["last_detection_time"]
            if time_diff < 60:
                time_str = f"{int(time_diff)}秒前"
            elif time_diff < 3600:
                time_str = f"{int(time_diff/60)}分钟前"
            else:
                time_str = f"{int(time_diff/3600)}小时前"
                
            detection_time = datetime.fromtimestamp(
                self.stats["last_detection_time"]).strftime("%H:%M:%S")
            last_time = f"{detection_time} ({time_str})"
        else:
            last_time = "--"
            
        # 更新最近检测结果
        if self.latest_results:
            if len(self.latest_results) > 0:
                result_str = f"发现{len(self.latest_results)}个缺陷"
                
                # 生成详情
                details = []
                for i, det in enumerate(self.latest_results):
                    if len(det) >= 6:
                        x1, y1, x2, y2, conf, class_id = det[:6]
                        details.append(f"{i+1}. 缺陷 (置信度: {conf:.2f})")
                        
                details_str = "\n".join(details)
            else:
                result_str = "未发现缺陷"
                details_str = "图像检测正常，未发现缺陷"
        else:
            result_str = "--"
            details_str = "暂无检测记录"
            
        # 更新统计信息
        self.ui_elements["-RUN-TIME-"].config(text=run_time_str)
        self.ui_elements["-TOTAL-INSPECTIONS-"].config(text=str(self.stats["total_inspections"]))
        self.ui_elements["-DEFECT-COUNT-"].config(text=str(self.stats["defect_count"]))
        self.ui_elements["-DEFECT-RATE-"].config(text=f"{defect_rate:.1f}%")
        
        # 更新缺陷统计
        defect_count = self.stats["defect_types"]["defect"]
        self.ui_elements["-DEFECT-COUNT-"].config(text=str(defect_count))
        
        # 更新进度条
        if "defect" in self.progress_vars:
            # 假设最多100个缺陷，超过则保持100%
            progress_value = min(defect_count, 100)
            self.progress_vars["defect"].set(progress_value)
                
        # 更新最近检测信息
        self.ui_elements["-LAST-DETECTION-TIME-"].config(text=last_time)
        self.ui_elements["-LAST-DETECTION-RESULT-"].config(text=result_str)
        
        # 更新详情文本框
        details_text = self.ui_elements["-DETECTION-DETAILS-"]
        details_text.config(state=tk.NORMAL)
        details_text.delete(1.0, tk.END)
        details_text.insert(tk.END, details_str)
        details_text.config(state=tk.DISABLED)
        
    def set_status(self, status: str, is_error: bool = False):
        """
        设置系统状态
        
        Args:
            status: 状态描述
            is_error: 是否为错误状态
        """
        self.system_status = status
        
        if self.status_var:
            self.status_var.set(status)
            if is_error:
                self.ui_elements["-STATUS-"].config(fg="red")
            else:
                self.ui_elements["-STATUS-"].config(fg="white")
    
    def start(self):
        """启动显示界面"""
        self.running = True
        
        # 启用/禁用按钮
        if self.root:
            self.ui_elements["-START-"].config(state=tk.DISABLED)
            self.ui_elements["-STOP-"].config(state=tk.NORMAL)
            self.set_status("检测运行中...")
    
    def stop(self):
        """停止显示界面"""
        if self.running:
            self.running = False
            
            # 启用/禁用按钮
            if self.root:
                self.ui_elements["-START-"].config(state=tk.NORMAL)
                self.ui_elements["-STOP-"].config(state=tk.DISABLED)
                self.set_status("检测已停止")
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_inspections": 0,
            "defect_count": 0,
            "start_time": time.time(),
            "last_detection_time": 0,
            "defect_types": {
                "defect": 0  # 只保留一种缺陷类型
            }
        }
        
        # 更新界面
        if self.root:
            self.ui_elements["-START-TIME-"].config(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            self._update_info()
            self.set_status("统计信息已重置")
    
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
                self._update_info()
                self.root.after(int(update_interval * 1000), update)
            
            # 启动定期更新
            update()
            
            # 主循环
            self.root.mainloop()
                
        except Exception as e:
            logger.error(f"显示界面运行出错: {e}")
        finally:
            if self.root:
                self.root.destroy()
                self.root = None
    
    def close(self):
        """关闭窗口"""
        if self.root:
            self.root.destroy()
            self.root = None


# 示例用法
# if __name__ == "__main__":
#     # 配置日志
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )
    
#     # 创建显示界面
#     display = DefectDisplay(window_title="瑕疵检测系统")
    
#     # 创建测试图像
#     test_img = np.zeros((480, 640, 3), dtype=np.uint8)
#     cv2.putText(test_img, "Loading...", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
#     # 更新初始图像
#     display.update_image(test_img)
    
#     # 启动界面
#     display.start()
    
#     # 模拟检测线程
#     def detection_thread():
#         import random
#         import time
#         import cv2
        
#         # 创建一些测试图像
#         test_images = []
#         for i in range(3):
#             img = np.zeros((480, 640, 3), dtype=np.uint8)
#             # 添加一些随机形状
#             for _ in range(10):
#                 x = random.randint(0, 600)
#                 y = random.randint(0, 440)
#                 color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#                 radius = random.randint(10, 50)
#                 cv2.circle(img, (x, y), radius, color, -1)
#             test_images.append(img)
            
#         while display.running:
#             time.sleep(2)  # 每2秒更新一次
            
#             if not display.running:
#                 break
                
#             # 随机选择一张图片
#             img = random.choice(test_images).copy()
            
#             # 随机生成一些检测结果
#             if random.random() < 0.7:  # 70%概率有缺陷
#                 num_defects = random.randint(1, 3)
#                 detections = []
                
#                 for _ in range(num_defects):
#                     x1 = random.randint(50, 550)
#                     y1 = random.randint(50, 350)
#                     w = random.randint(30, 100)
#                     h = random.randint(30, 100)
#                     conf = random.uniform(0.7, 0.98)
                    
#                     # 在图像上预先绘制检测框
#                     cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2)
#                     cv2.putText(img, f"Defect: {conf:.2f}", (x1, y1-5), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
#                     # 添加到检测结果列表
#                     detections.append([x1, y1, x1 + w, y1 + h, conf, 0])
                    
#                 # 更新图像和检测结果
#                 display.update_image(img, detections)
#                 display.set_status(f"检测完成，发现{len(detections)}个缺陷")
#             else:
#                 display.update_image(img, [])
#                 display.set_status("检测完成，未发现缺陷")
    
#     # 启动模拟检测线程
#     thread = threading.Thread(target=detection_thread, daemon=True)
#     thread.start()
    
#     # 运行主界面
#     display.run()
    
#     # 清理
#     display.close()