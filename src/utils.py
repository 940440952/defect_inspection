# 1. 日志设置
# 2. 配置文件加载
# 3. 时间戳生成与格式化
# 4. 路径处理
# 5. 异常处理实用函数
# -*- coding: utf-8 -*-
"""
Utility functions for defect inspection system.
Provides common functionality shared across modules.
"""

import os
import sys
import json
import time
import logging
import logging.handlers
from datetime import datetime
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np

def setup_logging(level: int = logging.INFO, 
                 log_file: Optional[str] = None, 
                 console: bool = True,
                 max_bytes: int = 10485760,  # 10MB
                 backup_count: int = 5) -> logging.Logger:
    """
    Configure logging for the application
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Path to log file, if None logs to console only
        console: Whether to output logs to console
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
    
    Returns:
        Logger object for the root logger
    """
    # Create main logger
    logger = logging.getLogger("DefectInspection")
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create handlers
    handlers = []
    
    # Add file handler if log file specified
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
            except OSError:
                print(f"Warning: Could not create log directory: {log_dir}")
                log_file = None
    
        if log_file:
            try:
                # Use rotating file handler to prevent logs from growing too large
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file, 
                    maxBytes=max_bytes, 
                    backupCount=backup_count
                )
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                handlers.append(file_handler)
            except (IOError, PermissionError) as e:
                print(f"Warning: Could not create log file: {e}")
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # Add all handlers to logger
    for handler in handlers:
        logger.addHandler(handler)
    
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    """
    从JSON文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    # 创建一个局部logger变量，避免全局变量问题
    local_logger = logging.getLogger("DefectInspection")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # 配置加载后进行验证
        if not isinstance(config, dict):
            local_logger.warning(f"配置文件格式错误: {config_path}, 预期为字典")
            return {}
            
        local_logger.debug(f"加载的配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
        return config
    except Exception as e:
        local_logger.error(f"加载配置文件失败 {config_path}: {str(e)}")
        return {}

def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get formatted timestamp string
    
    Args:
        format_str: Format string for datetime.strftime()
        
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime(format_str)

def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure that a directory exists, create if it doesn't
    
    Args:
        directory_path: Path to directory
        
    Returns:
        True if directory exists or was created, False otherwise
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            return True
        except OSError as e:
            print(f"Error creating directory {directory_path}: {e}")
            return False
    return True

def safe_json_serialize(obj: Any) -> Any:
    """
    Convert an object to a JSON serializable type
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON serializable version of the object
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: safe_json_serialize(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        # For custom objects
        return {key: safe_json_serialize(value) for key, value in obj.__dict__.items() 
                if not key.startswith('_')}
    else:
        # Default to string representation
        return str(obj)

def safe_file_write(file_path: str, content: str, mode: str = 'w') -> bool:
    """
    Safely write content to a file
    
    Args:
        file_path: Path to file
        content: Content to write
        mode: File open mode ('w' for write, 'a' for append)
        
    Returns:
        True if successful, False otherwise
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            print(f"Error creating directory {directory}: {e}")
            return False
    
    try:
        with open(file_path, mode) as f:
            f.write(content)
        return True
    except (IOError, PermissionError) as e:
        print(f"Error writing to file {file_path}: {e}")
        return False

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable form
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g. "2h 30m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    elif minutes > 0:
        return f"{minutes}m {secs:02d}s"
    else:
        return f"{secs}s"

def save_metadata(metadata: Dict[str, Any], file_path: str) -> bool:
    """
    Save metadata as JSON file
    
    Args:
        metadata: Dictionary containing metadata
        file_path: Path to save the metadata file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=safe_json_serialize)
        return True
    except Exception as e:
        print(f"Error saving metadata to {file_path}: {e}")
        return False

def calculate_statistics(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate derived statistics from raw statistics
    
    Args:
        stats: Dictionary containing raw statistics
        
    Returns:
        Dictionary with additional calculated statistics
    """
    result = stats.copy()
    
    # Calculate runtime
    if 'start_time' in stats:
        runtime = time.time() - stats['start_time']
        result['runtime'] = runtime
        result['runtime_formatted'] = format_duration(runtime)
    
    # Calculate defect rate
    if stats.get('total_inspections', 0) > 0:
        defect_rate = (stats.get('defects_found', 0) / stats['total_inspections']) * 100
        result['defect_rate'] = round(defect_rate, 2)
    else:
        result['defect_rate'] = 0.0
    
    # Calculate ejection rate
    if stats.get('total_inspections', 0) > 0:
        ejection_rate = (stats.get('ejections', 0) / stats['total_inspections']) * 100
        result['ejection_rate'] = round(ejection_rate, 2)
    else:
        result['ejection_rate'] = 0.0
    
    # Calculate throughput (units/hour)
    if 'start_time' in stats and stats.get('total_inspections', 0) > 0:
        runtime_hours = runtime / 3600
        if runtime_hours > 0:
            throughput = stats['total_inspections'] / runtime_hours
            result['throughput'] = round(throughput, 2)
    
    return result

def create_unique_filename(base_dir: str, prefix: str, suffix: str = '.jpg') -> str:
    """
    Create a unique filename with timestamp
    
    Args:
        base_dir: Base directory
        prefix: Filename prefix
        suffix: Filename suffix/extension
        
    Returns:
        Full path to unique filename
    """
    timestamp = get_timestamp()
    filename = f"{prefix}_{timestamp}{suffix}"
    return os.path.join(base_dir, filename)

def parse_size(size_str: str) -> Tuple[int, int]:
    """
    Parse a size string like '1920x1080' to (width, height)
    
    Args:
        size_str: Size string in format 'WIDTHxHEIGHT'
        
    Returns:
        Tuple of (width, height)
        
    Raises:
        ValueError: If string format is invalid
    """
    try:
        parts = size_str.split('x')
        if len(parts) != 2:
            raise ValueError(f"Invalid size format: {size_str}")
        
        width = int(parts[0])
        height = int(parts[1])
        return (width, height)
    except Exception as e:
        raise ValueError(f"Invalid size format '{size_str}': {e}")

# Additional imports that might be needed later
try:
    from datetime import date
except ImportError:
    pass


if __name__ == "__main__":
    # Test the utility functions
    logger = setup_logging(logging.DEBUG, log_file="logs/test_utils.log")
    logger.info("Testing utility functions")
    
    # Test loading config
    config_path = "../config.json"
    if os.path.exists(config_path):
        config = load_config(config_path)
        logger.info(f"Loaded configuration with {len(config)} settings")
    
    # Test timestamp
    logger.info(f"Current timestamp: {get_timestamp()}")
    
    # Test directory creation
    test_dir = "test_directory"
    if ensure_directory_exists(test_dir):
        logger.info(f"Directory {test_dir} exists or was created")
        # Clean up test directory
        try:
            os.rmdir(test_dir)
        except:
            pass
    
    # Test statistics calculation
    test_stats = {
        "start_time": time.time() - 3665,  # 1 hour, 1 minute, 5 seconds ago
        "total_inspections": 100,
        "defects_found": 15,
        "ejections": 12
    }
    
    calculated_stats = calculate_statistics(test_stats)
    logger.info(f"Calculated statistics: {calculated_stats}")
    
    logger.info("Utility tests completed")

def convert_grayscale_to_rgb(img: np.ndarray) -> np.ndarray:
    """
    将灰度图像转换为RGB/BGR图像
    
    Args:
        img: 输入的灰度图像 (H, W) 或 (H, W, 1)
        
    Returns:
        转换后的三通道图像 (H, W, 3)
    """
    if img is None:
        return None
        
    # 检查输入图像的维度
    shape = img.shape
    if len(shape) == 3 and shape[2] == 3:
        # 已经是三通道图像，直接返回
        return img
    
    # 确保图像是二维的（如果是三维但单通道，转为二维）
    if len(shape) == 3 and shape[2] == 1:
        img = img[:, :, 0]
    
    # 使用OpenCV将灰度图转换为BGR图像
    try:
        import cv2
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img_rgb
    except Exception:
        # 如果OpenCV方法失败，使用numpy复制通道
        h, w = img.shape
        img_rgb = np.zeros((h, w, 3), dtype=img.dtype)
        img_rgb[:, :, 0] = img
        img_rgb[:, :, 1] = img
        img_rgb[:, :, 2] = img
        return img_rgb