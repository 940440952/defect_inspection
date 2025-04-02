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
            return False
        except RequestException as e:
            logger.warning(f"Connection test failed: {str(e)}")
            return False
     
    def import_to_label_studio(self, image_path: str, detections: List = None, 
                           metadata: Optional[Dict] = None,
                           project_id: int = None) -> bool:
        """
        Import an image with pre-annotations to Label Studio for labeling
        
        Args:
            image_path: Path to the image file
            detections: Optional list of detections to use as pre-annotations
            metadata: Optional metadata to include with the task
            project_id: Label Studio project ID (overrides default project)
            
        Returns:
            Optional[Dict]: Task data if successful, None otherwise
        """
        if not self.api_url or not self.api_token:
            logger.warning("API not configured")
            return None
            
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
            
        try:
            # Use specified project ID or default
            active_project_id = project_id or self.project_id
            if not active_project_id:
                logger.error("No project ID specified or available")
                return None
            
            # Get image dimensions for coordinate conversion (needed for annotations)
            image_width = None
            image_height = None
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    image_width, image_height = img.size
            except Exception as e:
                logger.warning(f"Could not get image dimensions: {e}")
                # Continue anyway - we'll skip annotations if dimensions are missing
            
            # Step 1: Upload image file using multipart form data
            headers = self._get_headers()
            # Remove Content-Type from headers for file upload
            upload_headers = {k: v for k, v in headers.items() if k.lower() != 'content-type'}
            
            # Prepare file upload
            filename = os.path.basename(image_path)
            files = {
                'file': (filename, open(image_path, 'rb'), f"image/{self._get_image_mime(filename)}")
            }
            
            # Add metadata as form data
            form_data = {}
            if metadata:
                # Convert metadata to JSON string and add to form data
                form_data['metadata'] = json.dumps(metadata)
            
            # Send the request with file attachment
            try:
                response = requests.post(
                    f"{self.api_url}/api/projects/{active_project_id}/import?return_task_ids=true",
                    headers=upload_headers,
                    files=files,
                    data=form_data,
                    timeout=self.timeout * 2  # Increased timeout for file upload
                )
            finally:
                # Close the file handle
                files['file'][1].close()
            
            if response.status_code not in [200, 201]:
                logger.error(f"Failed to upload image: Status {response.status_code}")
                if response.text:
                    logger.error(f"Response: {response.text}")
                return None
                
            # Parse response to get task ID
            upload_result = response.json()
            logger.debug(f"Image upload response: {upload_result}")
            
            task_id = None
            if 'task_ids' in upload_result and upload_result['task_ids']:
                task_id = upload_result['task_ids'][0]
            
            if not task_id:
                logger.error("No task ID returned from image upload")
                return upload_result  # Return whatever we got
            
            logger.info(f"Successfully uploaded image to project {active_project_id}, task ID: {task_id}")
            
            # Step 2: If we have detections and dimensions, add annotations to the task
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
                if len(det) >= 6:  # x1, y1, x2, y2, conf, class_id
                    x, y, w, h, conf, class_id = det[:6]
                    
                    # Create result based on our labeling config
                    results.append({
                        "type": "rectangle",  # rectanglelabels for labeled boxes
                        "value": {
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                            "rotation": 0,
                        },
                        "to_name": "image",
                        "from_name": "rectangle"
                    })
            
            # Create annotation payload
            annotation_data = {
                "result": results
            }
            
            # Submit annotation
            headers = self._get_headers()
            response = requests.post(
                f"{self.api_url}/api/tasks/{task_id}/annotations/",
                headers=headers,
                json=annotation_data,
                timeout=self.timeout
            )
            
            if response.status_code not in [200, 201]:
                logger.error(f"Failed to add annotations to task: Status {response.status_code}")
                if response.text:
                    logger.error(f"Response: {response.text}")
                return False
                
            logger.info(f"Successfully added {len(results)} annotations to task {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding annotations to task: {e}")
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
#     # Configure basic logging
#     logging.basicConfig(
#         level=logging.DEBUG,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )
    
#     # 从配置中读取必要参数
#     config = {
#         "api_url": "http://192.168.110.154:8080",
#         "api_token": "f8806eb800e1655282702c53e609bbf1f261a996",
#         "line_name": "Line01",
#         "product_type": "Smartphone"
#     }

#     # 初始化 API 客户端
#     api_client = APIClient(
#         api_url=config["api_url"],
#         api_token=config["api_token"],
#         line_name=config["line_name"],
#         product_type=config["product_type"]
#     )

#     # API 客户端已经初始化完成，包括项目查询和创建
#     # 现在可以直接使用它来上传图像和标注

#     # 上传图像示例
#     image_path = "/home/gtm/defect_inspection/data/images/bus.jpg"
#     detections = [
#         [30, 40, 20, 30, 0.95, 0],  # scratch - x, y, width, height, confidence, class_id
#         [60, 80, 20, 20, 0.85, 1]   # dent
#     ]
#     metadata = {
#         "timestamp": time.time(),
#         "batch_id": "BATCH_2025_03_31",
#         "camera_id": "CAM_01"
#     }

#     result = api_client.import_to_label_studio(image_path, detections, metadata)
#     if result:
#         print(f"Successfully uploaded image with {len(detections)} detections")
#     else:
#         print("Failed to upload image")

    
#     api_client.close()