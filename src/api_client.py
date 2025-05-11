# -*- coding: utf-8 -*-
"""
API client module for defect inspection system.
Handles communication with Label Studio for data upload, annotation and management.
"""

import os
import json
import time
import logging
import threading
import requests
import base64
from urllib.parse import urljoin
from requests.exceptions import RequestException, ConnectionError, Timeout
from typing import Dict, List, Optional, Tuple, Union, Any

# Set up logging
logger = logging.getLogger("DefectInspection.APIClient")

class APIClient:
    def __init__(self, api_url: str, api_token: str = "", 
             line_name: str = "LineTest", product_type: str = "QC",
             timeout: int = 10, max_retries: int = 3, 
             retry_delay: int = 5):
        """
        Initialize API client with Label Studio as backend
        
        Args:
            api_url: Base URL for Label Studio instance
            api_token: API token for Label Studio access
            line_name: Production line name (used in project naming)
            product_type: Product type being inspected (used in project naming)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.api_url = api_url.rstrip('/') if api_url else ""
        self.api_token = api_token
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.upload_queue = []
        self.upload_lock = threading.Lock()
        self.upload_thread = None
        self.running = False
        self.project_id = None
        self.line_name = line_name
        self.product_type = product_type
        
        # Configure logging if not already configured
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Initialize connection if URL and token provided
        if self.api_url and self.api_token:
            try:
                logger.info(f"Connecting to Label Studio API at {self.api_url}")
                
                # Test connection
                if not self.test_connection():
                    logger.error("Failed to connect to Label Studio API")
                    return
                    
                logger.info("Connection to Label Studio successful")
                
                # List available projects
                try:
                    projects = self.get_projects()
                    project_count = projects.get('count', 0)
                    logger.info(f"Found {project_count} projects in Label Studio")
                except Exception as e:
                    logger.error(f"Failed to list projects: {e}")
                    projects = {'results': []}
                
                # Generate standard project name
                # Format: AUTO_{line_name}_{product_type}
                project_name = f"AUTO_{self.line_name}_{self.product_type}"
                logger.info(f"Using project name: {project_name}")
                
                # Find existing project or create new one
                project_id = None
                
                # First try to find existing project with this name
                for project in projects.get('results', []):
                    if project.get('title') == project_name:
                        project_id = project.get('id')
                        logger.info(f"Found existing project: {project_name} (ID: {project_id})")
                        break
                
                # Create project if not found
                if not project_id:
                    logger.info(f"Project '{project_name}' not found, creating new project")
                    project_id = self._find_or_create_project(project_name)

                if project_id:
                    self.project_id = project_id
                    self.set_active_project(project_id)
                    logger.info(f"Using project: {project_name} (ID: {self.project_id})")
                    
                    # Get project info to log task count
                    try:
                        project_info = self.get_project_info()
                        task_count = project_info.get('task_number', 0)
                        logger.info(f"Project has {task_count} tasks")
                    except Exception as e:
                        logger.warning(f"Could not get project info: {e}")
                else:
                    logger.error(f"Failed to initialize project: {project_name}")
            except Exception as e:
                logger.error(f"Failed to initialize connection: {e}")
        
        
        
        
    def _init_connection(self, project_name: str = "Defect Inspection") -> Optional[int]:
        """
        Initialize connection and ensure project exists
        
        Args:
            project_name: Name of the project to use or create
            
        Returns:
            Optional[int]: Project ID if successful, None otherwise
        """
        if not self.api_url or not self.api_token:
            return None
            
        try:
            # Check connection to API
            headers = self._get_headers()
            response = requests.get(
                f"{self.api_url}/api/projects/",
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to connect to Label Studio: Status {response.status_code}")
                return None
                
            # Find or create project
            project_id = self._find_or_create_project(project_name)
            return project_id
            
        except Exception as e:
            logger.error(f"Error initializing connection: {e}")
            return None
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get request headers with authentication
        
        Returns:
            Dict[str, str]: Headers dictionary
        """
        return {
            'Authorization': f'Token {self.api_token}',
            'Content-Type': 'application/json'
        }
            
    def _find_or_create_project(self, project_name: str) -> Optional[int]:
        """
        Find existing project by name or create a new one
        
        Args:
            project_name: Name of the project to find or create
            
        Returns:
            Optional[int]: Project ID if found or created, None otherwise
        """
        if not self.api_url or not self.api_token:
            return None
            
        try:
            # Get all projects
            headers = self._get_headers()
            response = requests.get(
                f"{self.api_url}/api/projects/",
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to list projects: Status {response.status_code}")
                return None
                
            # Parse response
            projects = response.json()
            
            for i, project in enumerate(projects.get('results')):
                if project.get('title') == project_name:
                    logger.info(f"Found existing project: {project_name} (ID: {project.get('id')})")
                    return project.get('id')
            
            # Create new project if not found
            logger.info(f"Creating new project: {project_name}")
            
            # Define labeling config for object detection
            labeling_config = """
            <View>
                <Image name="image" value="$image"/>
                <Rectangle name="rectangle" toName="image"/>  <!-- 使用Rectangle而不是RectangleLabels -->
            </View>
            """
            
            # Create project
            create_data = {
                'title': project_name,
                'description': 'Automatically created by Defect Inspection System',
                'label_config': labeling_config
            }
            
            response = requests.post(
                f"{self.api_url}/api/projects/",
                headers=headers,
                json=create_data,
                timeout=self.timeout
            )
            
            if response.status_code not in [201, 200]:
                logger.error(f"Failed to create project: Status {response.status_code}")
                return None
                
            # Get project ID from response
            project_data = response.json()
            project_id = project_data.get('id')
            
            if project_id:
                logger.info(f"Created new project: {project_name} (ID: {project_id})")
                return project_id
            else:
                logger.error("Project created but no ID returned")
                return None
                
        except Exception as e:
            logger.error(f"Error finding/creating project: {e}")
            return None
                
    def test_connection(self) -> bool:
        """
        Test API connection and authentication
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not self.api_url:
            logger.warning("No API URL configured, skipping connection test")
            return False
            
        try:
            headers = self._get_headers()
            response = requests.get(
                f"{self.api_url}/api/projects/",
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                logger.info("Connection test successful")
                return True
            elif response.status_code == 401:
                logger.error("Authentication failed (invalid token)")
                return False
            else:
                logger.warning(f"Connection test returned status {response.status_code}")
                return False
                
        except ConnectionError:
            logger.warning("Connection test failed - server unreachable")
            # 打印ConnectionError
            print(ConnectionError)
            return False
        except RequestException as e:
            logger.warning(f"Connection test failed: {str(e)}")
            return False
     
    def import_to_label_studio(self, image, detections: List = None, 
                       metadata: Optional[Dict] = None,
                       project_id: int = None,
                       is_path: bool = True) -> bool:
        """
        Import an image with pre-annotations to Label Studio for labeling
        
        Args:
            image: Path to the image file or numpy array containing image data
            detections: Optional list of detections to use as pre-annotations
            metadata: Optional metadata to include with the task
            project_id: Label Studio project ID (overrides default project)
            is_path: If True, 'image' is a file path; if False, 'image' is a numpy array
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.api_url or not self.api_token:
            logger.warning("API not configured")
            return False
            
        try:
            # Use specified project ID or default
            active_project_id = project_id or self.project_id
            
            if not active_project_id:
                logger.error("No project ID specified or available")
                return False
            
            # 处理图像 - 根据is_path参数判断输入类型
            if is_path:
                # 检查文件是否存在
                if not os.path.exists(image):
                    logger.error(f"Image file not found: {image}")
                    return False
                    
                # 获取图像尺寸
                try:
                    from PIL import Image as PILImage
                    with PILImage.open(image) as img:
                        image_width, image_height = img.size
                except Exception as e:
                    logger.warning(f"Could not get image dimensions: {e}")
                    # 继续执行 - 如果无法获取尺寸，将跳过添加标注
                
                # 准备文件上传
                filename = os.path.basename(image)
                file_data = open(image, 'rb')
            else:
                # 处理numpy数组
                import cv2
                import numpy as np
                from io import BytesIO
                
                if not isinstance(image, np.ndarray):
                    logger.error("Invalid image data: expected numpy array")
                    return False
                    
                # 从数组获取尺寸
                image_height, image_width = image.shape[:2]
                
                # 转换为JPEG格式
                _, buffer = cv2.imencode(".jpg", image)
                file_data = BytesIO(buffer.tobytes())
                filename = f"image_{int(time.time())}.jpg"
            
            # 准备上传
            headers = self._get_headers()
            # 移除Content-Type以便文件上传
            upload_headers = {k: v for k, v in headers.items() if k.lower() != 'content-type'}
            
            # 准备文件上传
            files = {
                'file': (filename, file_data, f"image/{self._get_image_mime(filename)}")
            }
            
            # 添加元数据
            form_data = {}
            if metadata:
                form_data['metadata'] = json.dumps(metadata)
            
            # 发送请求
            try:
                response = requests.post(
                    f"{self.api_url}/api/projects/{active_project_id}/import?return_task_ids=true",
                    headers=upload_headers,
                    files=files,
                    data=form_data,
                    timeout=self.timeout * 2
                )
            finally:
                # 关闭文件
                file_data.close()
            
            if response.status_code not in [200, 201]:
                logger.error(f"Failed to upload image: Status {response.status_code}")
                if response.text:
                    logger.error(f"Response: {response.text}")
                return False
                
            # 解析响应获取task_id
            upload_result = response.json()
            logger.debug(f"Image upload response: {upload_result}")
            
            task_id = None
            if 'task_ids' in upload_result and upload_result['task_ids']:
                task_id = upload_result['task_ids'][0]
            
            if not task_id:
                logger.error("No task ID returned from image upload")
                return False
            
            logger.info(f"Successfully uploaded image to project {active_project_id}, task ID: {task_id}")
            
            # 如果有检测结果和图像尺寸，添加标注
            if detections and image_width and image_height:
                annotation_result = self._add_annotations_to_task(
                    task_id, detections, image_width, image_height
                )
                if not annotation_result:
                    logger.warning("Failed to add annotations to task")
            
            return True
                
        except Exception as e:
            logger.error(f"Error importing to Label Studio: {e}")
            return False

    def _add_annotations_to_task(self, task_id: int, detections: List, 
                           image_width: int, image_height: int) -> bool:
        """
        Add annotations to an existing task
        
        Args:
            task_id: Task ID to add annotations to
            detections: List of detection results (format: [x1, y1, x2, y2, conf, class_id])
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not detections:
            return True  # Nothing to add
            
        try:
            results = []
            
            # Create annotation in Label Studio format
            for i, det in enumerate(detections):
                if len(det) >= 6:  # x, y, width, height, conf, class_id
                    # 确保所有数值都转换为Python原生类型，而不是numpy类型
                    x1 = float(det[0])
                    y1 = float(det[1])
                    x2 = float(det[2])
                    y2 = float(det[3])
                    conf = float(det[4])
                    class_id = int(det[5])
                    
                    # 计算相对宽度和高度
                    width = (x2 - x1) / image_width * 100
                    height = (y2 - y1) / image_height * 100
                    
                    # 转换为相对坐标 (0-100%)
                    x = x1 / image_width * 100
                    y = y1 / image_height * 100
                    
                    # 创建Label Studio格式的标注
                    results.append({
                        "type": "rectangle",
                        "value": {
                            "x": float(x),
                            "y": float(y),
                            "width": float(width),
                            "height": float(height),
                            "rotation": 0,
                        },
                        "to_name": "image",
                        "from_name": "rectangle"
                    })
            
            # 确保所有数据都是JSON可序列化的
            annotation_data = {
                "result": results,
                "score": 1.0,
                "ground_truth": False,
                "lead_time": 0
            }
            
            # 提交标注
            headers = self._get_headers()
            response = requests.post(
                f"{self.api_url}/api/tasks/{task_id}/annotations/",
                headers=headers,
                json=annotation_data,  # 使用json参数会自动处理序列化
                timeout=self.timeout
            )
            
            if response.status_code not in [200, 201]:
                logger.error(f"添加标注失败: 状态码 {response.status_code}")
                if response.text:
                    logger.error(f"响应: {response.text}")
                return False
                
            logger.info(f"成功为任务 {task_id} 添加 {len(results)} 个标注")
            return True
            
        except Exception as e:
            logger.error(f"添加标注时出错: {e}")
            return False
            

    def _get_image_mime(self, filename: str) -> str:
        """
        Get the MIME type for an image based on file extension
        
        Args:
            filename: Image filename
            
        Returns:
            str: MIME type string
        """
        ext = os.path.splitext(filename.lower())[1]
        if ext == '.jpg' or ext == '.jpeg':
            return 'jpeg'
        elif ext == '.png':
            return 'png'
        elif ext == '.gif':
            return 'gif'
        elif ext == '.bmp':
            return 'bmp'
        elif ext == '.webp':
            return 'webp'
        else:
            return 'jpeg'  # Default to JPEG
    
    def get_projects(self) -> Dict:
        """
        Get list of projects from Label Studio
        
        Returns:
            List[Dict]: List of project dictionaries
        """
        if not self.api_url or not self.api_token:
            logger.warning("API not configured")
            return {}
            
        try:
            headers = self._get_headers()
            response = requests.get(
                f"{self.api_url}/api/projects/",
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get projects: Status {response.status_code}")
                return {}
                
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting projects: {e}")
            return {}
            
    def set_active_project(self, project_id: int) -> bool:
        """
        Set active project ID
        
        Args:
            project_id: Project ID to set as active
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.api_url or not self.api_token:
            logger.warning("API not configured")
            return False
            
        try:
            # Check if project exists
            headers = self._get_headers()
            response = requests.get(
                f"{self.api_url}/api/projects/{project_id}/",
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.error(f"Project {project_id} not found: Status {response.status_code}")
                return False
                
            # Set as active project
            self.project_id = project_id
            logger.info(f"Set active project: {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting active project: {e}")
            return False
    
    def get_project_info(self, project_id: Optional[int] = None, filters: Optional[Dict] = None, 
                 page: int = 1, page_size: int = 100) -> Dict:
        """
        Get tasks from a project with optional filtering
        
        Args:
            project_id: Project ID (uses default if None)
            filters: Optional filters to apply
            page: Page number for pagination
            page_size: Number of items per page
            
        Returns:
            List[Dict]: List of task dictionaries
        """
        if not self.api_url or not self.api_token:
            logger.warning("API not configured")
            return {}
            
        try:
            active_project_id = project_id or self.project_id
            if not active_project_id:
                logger.error("No project ID specified or available")
                return {}
                
            headers = self._get_headers()
            
            
            response = requests.get(
                f"{self.api_url}/api/projects/{active_project_id}",
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get tasks: Status {response.status_code}")
                return {}
                
            result = response.json()
            return result
            
        except Exception as e:
            logger.error(f"Error getting tasks: {e}")
            return {}
    
    def get_task_annotations(self, task_id: int) -> List[Dict]:
        """
        Get annotations for a specific task
        
        Args:
            task_id: Task ID to get annotations for
            
        Returns:
            List[Dict]: List of annotation dictionaries
        """
        if not self.api_url or not self.api_token:
            logger.warning("API not configured")
            return []
            
        try:
            headers = self._get_headers()
            response = requests.get(
                f"{self.api_url}/api/tasks/{task_id}/annotations/",
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get annotations: Status {response.status_code}")
                return []
                
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting annotations: {e}")
            return []
    
    def get_system_status(self) -> Optional[Dict]:
        """
        Get system status (custom endpoint or Label Studio health check)
        
        Returns:
            Optional[Dict]: Status information or None if failed
        """
        if not self.api_url:
            return None
            
        try:
            headers = self._get_headers()
            
            # Try Label Studio health endpoint
            response = requests.get(
                f"{self.api_url}/health",
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                # Get active project info if available
                status_info = {
                    "status": "online",
                    "version": "Label Studio",
                }
                
                if self.project_id:
                    try:
                        project_response = requests.get(
                            f"{self.api_url}/api/projects/{self.project_id}/",
                            headers=headers,
                            timeout=self.timeout
                        )
                        if project_response.status_code == 200:
                            project_data = project_response.json()
                            status_info["project"] = project_data.get("title", "Unknown")
                            status_info["project_id"] = self.project_id
                            status_info["task_count"] = project_data.get("task_count", 0)
                    except:
                        pass
                
                return status_info
            else:
                logger.warning(f"Failed to get system status: Status {response.status_code}")
                return None
                
        except RequestException as e:
            logger.warning(f"Error getting system status: {str(e)}")
            return None
    
    def get_task_predictions(self, task_id: int) -> List[Dict]:
        """
        Get predictions for a specific task
        
        Args:
            task_id: Task ID to get predictions for
            
        Returns:
            List[Dict]: List of prediction dictionaries
        """
        if not self.api_url or not self.api_token:
            logger.warning("API not configured")
            return []
            
        try:
            headers = self._get_headers()
            response = requests.get(
                f"{self.api_url}/api/tasks/{task_id}/predictions/",
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get predictions: Status {response.status_code}")
                return []
                
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return []
        
    def close(self) -> None:
        """Clean up resources"""
        self.running = False
        logger.debug("API client closed")


# test
# if __name__ == "__main__":
#     # 配置日志
#     logging.basicConfig(
#         level=logging.DEBUG,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )

#     # 从配置中读取必要参数
#     config = {
#         "api_url": "http://192.168.4.53:8080",
#         "api_token": "ed1c8c53f2eb220ca6a324e5e166b99eea33251a",
#         "line_name": "Line01",
#         "product_type": "盖子"
#     }

#     print("-" * 50)
#     print("测试API客户端")
#     print("-" * 50)
    
#     # 初始化API客户端
#     print("初始化API客户端...")
#     api_client = APIClient(
#         api_url=config["api_url"],
#         api_token=config["api_token"],
#         line_name=config["line_name"],
#         product_type=config["product_type"]
#     )

#     # 测试连接
#     print(f"连接状态: {'成功' if api_client.test_connection() else '失败'}")
#     print(f"当前项目ID: {api_client.project_id}")
    
#     # 测试两种方式上传图像
#     import numpy as np
#     import cv2
#     # 1. 测试文件路径上传
#     print("\n测试 #1: 使用文件路径上传")
    
#     # 创建测试图像目录
#     test_dir = "/home/gtm/defect_inspection/data/images"
#     os.makedirs(test_dir, exist_ok=True)
    
#     # 创建测试图像文件
#     image_path = os.path.join(test_dir, "test_image.jpg")
#     test_img1 = np.zeros((480, 640, 3), dtype=np.uint8)
#     cv2.rectangle(test_img1, (100, 100), (300, 300), (0, 0, 255), 2)
#     cv2.putText(test_img1, "Test Image", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#     cv2.imwrite(image_path, test_img1)
#     print(f"创建测试图像: {image_path}")
    
#     # 模拟检测结果
#     detections1 = [
#         [80, 60, 30, 10, 0.95, 0],  # x, y, width, height, confidence, class_id
#     ]
    
#     metadata1 = {
#         "timestamp": time.time(),
#         "batch_id": "TEST_BATCH_1",
#         "camera_id": "CAM_TEST",
#         "test_type": "file_path"
#     }

#     # 使用文件路径上传
#     result1 = api_client.import_to_label_studio(image_path, detections1, metadata1, is_path=True)
#     print(f"文件路径上传结果: {'成功' if result1 else '失败'}")

#     # 2. 测试numpy数组上传
#     print("\n测试 #2: 使用numpy数组上传")
    

#     # 创建测试图像
#     test_img2 = np.zeros((480, 640, 3), dtype=np.uint8)
#     cv2.rectangle(test_img2, (200, 150), (400, 350), (0, 255, 0), 2)
#     cv2.putText(test_img2, "Numpy Image", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
#     # 模拟检测结果
#     detections2 = [
#         [30, 60, 20, 50, 0.85, 0],  # x, y, width, height, confidence, class_id
#     ]
    
#     metadata2 = {
#         "timestamp": time.time(),
#         "batch_id": "TEST_BATCH_2",
#         "camera_id": "CAM_TEST",
#         "test_type": "numpy_array"
#     }

#     # 使用numpy数组上传
#     result2 = api_client.import_to_label_studio(test_img2, detections2, metadata2, is_path=False)
#     print(f"numpy数组上传结果: {'成功' if result2 else '失败'}")

#     # 清理
#     api_client.close()
#     print("\n测试完成")
#     print("-" * 50)