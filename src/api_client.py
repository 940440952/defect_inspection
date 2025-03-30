# -*- coding: utf-8 -*-
"""
API client module for defect inspection system.
Handles communication with remote servers for data upload and synchronization.
Also supports integration with Label Studio for data annotation.
"""

import os
import json
import time
import logging
import threading
import requests
from urllib.parse import urljoin
from requests.exceptions import RequestException, ConnectionError, Timeout
from typing import Dict, List, Optional, Tuple, Union, Any

# Set up logging
logger = logging.getLogger("DefectInspection.APIClient")

class APIClient:
    """
    API client for uploading detection results and images to remote server.
    Also supports Label Studio integration for data labeling workflows.
    """
    
    def __init__(self, api_url: str, auth_token: str = "", 
                 timeout: int = 10, max_retries: int = 3, 
                 retry_delay: int = 5, label_studio_url: str = None,
                 label_studio_token: str = None):
        """
        Initialize API client
        
        Args:
            api_url: Base URL for the API
            auth_token: Authentication token for API access
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            label_studio_url: URL for Label Studio instance (if used)
            label_studio_token: API token for Label Studio (if used)
        """
        self.api_url = api_url.rstrip('/') if api_url else ""
        self.auth_token = auth_token
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.upload_queue = []
        self.upload_lock = threading.Lock()
        self.upload_thread = None
        self.running = False
        
        # Label Studio configuration
        self.label_studio_url = label_studio_url
        self.label_studio_token = label_studio_token
        self.label_studio_client = None
        
        # Initialize Label Studio client if configured
        if label_studio_url and label_studio_token:
            try:
                self._init_label_studio_client()
            except ImportError:
                logger.warning("Label Studio SDK not installed. Run 'pip install label-studio-sdk' to enable this feature.")
            except Exception as e:
                logger.error(f"Failed to initialize Label Studio client: {e}")
        
        # Test API connection if URL provided
        if self.api_url:
            self.test_connection()
            logger.info(f"API client initialized with endpoint: {self.api_url}")
        
    def _init_label_studio_client(self):
        """Initialize Label Studio client if SDK is available"""
        try:
            from label_studio_sdk import Client
            self.label_studio_client = Client(
                url=self.label_studio_url,
                api_key=self.label_studio_token
            )
            logger.info(f"Label Studio client initialized: {self.label_studio_url}")
        except Exception as e:
            logger.error(f"Error initializing Label Studio client: {e}")
            self.label_studio_client = None
                
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
                urljoin(self.api_url, "/status"),
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                logger.info("API connection test successful")
                return True
            elif response.status_code == 401:
                logger.error("API authentication failed (invalid token)")
                return False
            else:
                logger.warning(f"API connection test returned status {response.status_code}")
                return False
                
        except ConnectionError:
            logger.warning("API connection test failed - server unreachable")
            return False
        except RequestException as e:
            logger.warning(f"API connection test failed: {str(e)}")
            return False
            
    def _get_headers(self) -> Dict[str, str]:
        """
        Get request headers with authentication
        
        Returns:
            Dict[str, str]: Headers dictionary
        """
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'DefectInspection/1.0',
        }
        
        if self.auth_token:
            headers['Authorization'] = f'Bearer {self.auth_token}'
            
        return headers
        
    def upload_result(self, image_path: str, detections: List, 
                      metadata: Optional[Dict] = None) -> bool:
        """
        Upload detection result and image to API server
        
        Args:
            image_path: Path to the image file
            detections: List of detection results (format: [x1, y1, x2, y2, conf, class_id])
            metadata: Additional metadata to include
            
        Returns:
            bool: True if upload was successful (or queued), False on error
        """
        if not self.api_url:
            logger.debug("No API URL configured, skipping upload")
            return False
            
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return False
            
        # Prepare data
        data = {
            'timestamp': time.time(),
            'image_filename': os.path.basename(image_path),
            'detections': []
        }
        
        # Add detections to data
        for det in detections:
            if len(det) >= 6:  # x1, y1, x2, y2, conf, class_id
                x1, y1, x2, y2, conf, class_id = det[:6]
                detection_data = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class_id': int(class_id)
                }
                
                # Add class name if available in metadata
                if metadata and 'class_names' in metadata and int(class_id) < len(metadata['class_names']):
                    detection_data['class_name'] = metadata['class_names'][int(class_id)]
                    
                data['detections'].append(detection_data)
        
        # Add metadata if provided
        if metadata:
            data['metadata'] = metadata
            
        # Upload in background to avoid blocking
        try:
            return self._upload_data(image_path, data)
        except Exception as e:
            logger.error(f"Error starting upload: {str(e)}")
            return False
            
    def _upload_data(self, image_path: str, data: Dict) -> bool:
        """
        Perform the actual upload with retries
        
        Args:
            image_path: Path to image file
            data: JSON data to upload
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.api_url:
            return False
            
        # Prepare multipart form data
        files = {'image': (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')}
        payload = {'data': json.dumps(data)}
        
        # Try to upload with retries
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    urljoin(self.api_url, "/upload"),
                    files=files,
                    data=payload,
                    headers=self._get_headers(),
                    timeout=self.timeout
                )
                
                # Close the file
                files['image'][1].close()
                
                if response.status_code in [200, 201]:
                    logger.info(f"Upload successful: {os.path.basename(image_path)}")
                    return True
                else:
                    logger.warning(f"Upload failed (attempt {attempt}/{self.max_retries}): "
                                  f"Status code {response.status_code}")
                    
            except (ConnectionError, Timeout) as e:
                logger.warning(f"Upload connection error (attempt {attempt}/{self.max_retries}): {str(e)}")
            except RequestException as e:
                logger.warning(f"Upload request error (attempt {attempt}/{self.max_retries}): {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error during upload: {str(e)}")
                try:
                    files['image'][1].close()
                except:
                    pass
                return False
                
            # Delay before retry (if not last attempt)
            if attempt < self.max_retries:
                time.sleep(self.retry_delay)
                
        # Close file if still open after all retries
        try:
            if not files['image'][1].closed:
                files['image'][1].close()
        except:
            pass
            
        logger.error(f"Upload failed after {self.max_retries} attempts: {os.path.basename(image_path)}")
        return False
        
    def upload_batch(self, batch_data: List[Dict]) -> bool:
        """
        Upload a batch of results without images
        
        Args:
            batch_data: List of result dictionaries
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.api_url:
            logger.debug("No API URL configured, skipping batch upload")
            return False
            
        headers = self._get_headers()
        
        try:
            response = requests.post(
                urljoin(self.api_url, "/batch"),
                json={'batch': batch_data},
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"Batch upload successful: {len(batch_data)} items")
                return True
            else:
                logger.error(f"Batch upload failed: Status code {response.status_code}")
                return False
                
        except RequestException as e:
            logger.error(f"Batch upload error: {str(e)}")
            return False

    def get_system_status(self) -> Optional[Dict]:
        """
        Get system status from API
        
        Returns:
            Optional[Dict]: Status information or None if failed
        """
        if not self.api_url:
            return None
            
        try:
            headers = self._get_headers()
            response = requests.get(
                urljoin(self.api_url, "/system/status"),
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get system status: Status code {response.status_code}")
                return None
                
        except RequestException as e:
            logger.warning(f"Error getting system status: {str(e)}")
            return None
    
    def import_to_label_studio(self, image_path: str, detections: List = None, 
                              project_id: int = None) -> bool:
        """
        Import an image with pre-annotations to Label Studio for labeling
        
        Args:
            image_path: Path to the image file
            detections: Optional list of detections to use as pre-annotations
            project_id: Label Studio project ID (required if multiple projects exist)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.label_studio_client:
            logger.warning("Label Studio client not initialized")
            return False
            
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return False
            
        try:
            # Find project if ID not specified
            if not project_id:
                projects = self.label_studio_client.list_projects()
                if not projects:
                    logger.error("No projects found in Label Studio")
                    return False
                project_id = projects[0].id
                logger.debug(f"Using project ID {project_id}")
                
            # Get project object
            project = self.label_studio_client.get_project(project_id)
            
            # Create task with pre-annotations if detections provided
            filename = os.path.basename(image_path)
            task_data = {
                'image': open(image_path, 'rb'),
                'filename': filename,
            }
            
            # Add pre-annotations if provided
            annotations = []
            if detections:
                results = []
                # Create annotation in Label Studio format
                for i, det in enumerate(detections):
                    if len(det) >= 6:  # x1, y1, x2, y2, conf, class_id
                        x1, y1, x2, y2, conf, class_id = det[:6]
                        
                        # Convert to relative coordinates that Label Studio expects
                        # This assumes we know the image dimensions - would need to get this from the image
                        from PIL import Image
                        img = Image.open(image_path)
                        img_width, img_height = img.size
                        img.close()
                        
                        # Convert to normalized coordinates (0-100%)
                        x1_norm = x1 / img_width * 100
                        y1_norm = y1 / img_height * 100
                        width_norm = (x2 - x1) / img_width * 100
                        height_norm = (y2 - y1) / img_height * 100
                        
                        # Create result in Label Studio format
                        results.append({
                            "id": f"result_{i+1}",
                            "type": "rectanglelabels",
                            "value": {
                                "x": float(x1_norm),
                                "y": float(y1_norm),
                                "width": float(width_norm),
                                "height": float(height_norm),
                                "rotation": 0,
                                "rectanglelabels": [f"class_{int(class_id)}"]
                            },
                            "to_name": "image",
                            "from_name": "label"
                        })
                
                # Create pre-annotation
                if results:
                    annotations = [{
                        "result": results,
                        "ground_truth": False,
                        "lead_time": 0,
                        "was_cancelled": False,
                        "task": None  # Will be set by Label Studio
                    }]
                
            # Create import storage
            import_storage = project.create_import_storage("local_files")
            
            # Import file
            import_storage.upload_file(image_path)
            
            # Connect storage to project
            import_storage.sync()
            
            logger.info(f"Successfully imported {filename} to Label Studio project {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing to Label Studio: {e}")
            return False

    def close(self) -> None:
        """Clean up resources"""
        self.running = False
        logger.debug("API client closed")