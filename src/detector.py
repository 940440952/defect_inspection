# -*- coding: utf-8 -*-
"""
YOLO detector module for defect inspection system.
Supports YOLOv11L models with ONNX runtime acceleration.
"""

import os
import cv2
import time
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Union

# Set up logging
logger = logging.getLogger("DefectInspection.Detector")

class YOLODetector:
    """YOLO object detector with support for YOLOv11L models"""
    
    def __init__(self, model_path: str, conf_thresh: float = 0.25,
                 nms_thresh: float = 0.45, img_size: int = 640,
                 use_cuda: bool = True):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to ONNX model file
            conf_thresh: Confidence threshold for detections
            nms_thresh: NMS IoU threshold
            img_size: Input image size for model
            use_cuda: Whether to use CUDA acceleration if available
        """
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.img_size = img_size
        self.use_cuda = use_cuda
        self.session = None
        self.input_name = None
        self.output_names = None
        self.class_names = ["scratch", "dent", "stain", "crack", "deformation"]
        
        # Load model
        self._load_model()
        logger.info(f"YOLOv11L detector initialized with model: {os.path.basename(model_path)}")
        logger.debug(f"Input size: {self.img_size}, Confidence threshold: {self.conf_thresh}")
        
    def _load_model(self):
        """Load ONNX model and create inference session"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            import onnxruntime as ort
            
            # Set up ONNX runtime session
            providers = []
            if self.use_cuda and 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append('CUDAExecutionProvider')
                logger.info("Using CUDA for inference")
            else:
                if self.use_cuda:
                    logger.warning("CUDA requested but not available, falling back to CPU")
                providers.append('CPUExecutionProvider')
                logger.info("Using CPU for inference")
            
            # Create session with selected execution providers
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get input and output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # Run a warmup inference to initialize the model
            dummy_input = np.zeros((1, 3, self.img_size, self.img_size), dtype=np.float32)
            self.session.run(self.output_names, {self.input_name: dummy_input})
            logger.debug("Model loaded and warmup inference completed")
            
        except ImportError:
            logger.error("Failed to import onnxruntime. Please install with: pip install onnxruntime-gpu")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image for inference
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            tuple: (preprocessed_img, ratio, (padding_w, padding_h))
        """
        # Get original image dimensions
        h, w = img.shape[:2]
        
        # Calculate resize ratio and padding
        ratio = min(self.img_size / w, self.img_size / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        pad_w, pad_h = (self.img_size - new_w) // 2, (self.img_size - new_h) // 2
        
        # Resize image
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        # Convert to RGB and normalize to 0-1
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        padded = padded.astype(np.float32) / 255.0
        
        # HWC to NCHW (batch, channels, height, width)
        padded = padded.transpose(2, 0, 1)
        padded = np.expand_dims(padded, axis=0)
        
        return padded, ratio, (pad_w, pad_h)
    
    def post_process(self, outputs: List[np.ndarray], 
                    ratio: float, 
                    pad: Tuple[int, int]) -> List[List[float]]:
        """
        Process raw model output
        
        Args:
            outputs: Model outputs
            ratio: Resize ratio
            pad: Padding (w, h)
            
        Returns:
            List of detections [x1, y1, x2, y2, conf, class_id]
        """
        # Extract predictions from output
        if len(outputs) == 1:
            # YOLOv11L standard output format
            predictions = outputs[0]
        elif len(outputs) >= 4:
            # Legacy format
            predictions = np.concatenate([o for o in outputs], axis=1)
        else:
            logger.error(f"Unsupported output format with {len(outputs)} outputs")
            return []
        
        # Filter by confidence
        scores = predictions[:, 4:5] * predictions[:, 5:]  # conf * class_prob
        mask = scores.max(1) >= self.conf_thresh
        predictions = predictions[mask]
        
        if len(predictions) == 0:
            return []
            
        # Get class scores and IDs
        class_scores = scores[mask]
        class_ids = class_scores.argmax(1, keepdims=True)
        
        # Get bounding boxes
        boxes = predictions[:, :4]
        
        # Remove padding and rescale boxes to original image size
        pad_w, pad_h = pad
        boxes[:, 0] = (boxes[:, 0] - pad_w) / ratio  # x1
        boxes[:, 1] = (boxes[:, 1] - pad_h) / ratio  # y1
        boxes[:, 2] = (boxes[:, 2] - pad_w) / ratio  # x2
        boxes[:, 3] = (boxes[:, 3] - pad_h) / ratio  # y2
        
        # Clip boxes to image bounds
        # boxes[:, 0].clip(0, img_w)
        # boxes[:, 1].clip(0, img_h)
        # boxes[:, 2].clip(0, img_w)
        # boxes[:, 3].clip(0, img_h)
        
        # Combine detections
        max_scores = class_scores.max(1)
        detections = np.concatenate([boxes, max_scores.reshape(-1, 1), class_ids.astype(np.float32)], axis=1)
        
        # Non-maximum suppression
        keep = self._nms(detections[:, :4], detections[:, 4])
        detections = detections[keep]
        
        return detections.tolist()
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> List[int]:
        """
        Non-maximum suppression
        
        Args:
            boxes: Bounding boxes [N, 4]
            scores: Confidence scores [N]
            
        Returns:
            List of indices to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
            
        return keep
    
    def detect(self, image: np.ndarray) -> List[List[float]]:
        """
        Detect objects in image
        
        Args:
            image: Input image in BGR format (HWC)
            
        Returns:
            List of detections [x1, y1, x2, y2, conf, class_id]
        """
        if image is None or not isinstance(image, np.ndarray):
            logger.error("Invalid input image")
            return []
            
        try:
            # Preprocessing
            img_input, ratio, pad = self.preprocess(image.copy())
            
            # Inference
            start_time = time.time()
            outputs = self.session.run(self.output_names, {self.input_name: img_input})
            inference_time = (time.time() - start_time) * 1000
            logger.debug(f"Inference time: {inference_time:.2f}ms")
            
            # Postprocessing
            detections = self.post_process(outputs, ratio, pad)
            
            return detections
            
        except Exception as e:
            logger.exception(f"Error during detection: {e}")
            return []
    
    def draw_detections(self, image: np.ndarray, detections: List[List[float]]) -> np.ndarray:
        """
        Draw detection bounding boxes on image
        
        Args:
            image: Input image
            detections: List of detections [x1, y1, x2, y2, conf, class_id]
            
        Returns:
            Image with drawn detections
        """
        img = image.copy()
        
        # Define colors for each class
        colors = [
            (0, 0, 255),     # Red for scratch
            (0, 255, 255),   # Yellow for dent
            (0, 128, 255),   # Orange for stain
            (255, 0, 0),     # Blue for crack
            (128, 0, 255),   # Purple for deformation
        ]
        
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            
            # Convert to int
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(class_id)
            
            # Get color for this class
            color = colors[class_id % len(colors)]
            
            # Get class name
            if class_id < len(self.class_names):
                class_name = self.class_names[class_id]
            else:
                class_name = f"class_{class_id}"
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img


# For testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test model path - update this to your actual model path
    model_path = "models/yolov11l_defect.onnx"
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        exit(1)
    
    # Test with local image if available
    test_img_path = "test_image.jpg"
    if not os.path.exists(test_img_path):
        logger.error(f"Test image not found: {test_img_path}")
        exit(1)
        
    try:
        # Initialize detector
        detector = YOLODetector(model_path, conf_thresh=0.25, use_cuda=True)
        
        # Load and process image
        img = cv2.imread(test_img_path)
        logger.info(f"Loaded image with shape: {img.shape}")
        
        # Perform detection
        start_time = time.time()
        detections = detector.detect(img)
        elapsed = (time.time() - start_time) * 1000
        
        # Log results
        logger.info(f"Detection completed in {elapsed:.2f}ms")
        logger.info(f"Found {len(detections)} objects")
        
        # Draw detections
        result_img = detector.draw_detections(img, detections)
        
        # Save and display result
        output_path = "detection_result.jpg"
        cv2.imwrite(output_path, result_img)
        logger.info(f"Result saved to {output_path}")
        
        # Display if running in interactive environment
        cv2.imshow("Detection Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        logger.exception(f"Error during test: {e}")
    
