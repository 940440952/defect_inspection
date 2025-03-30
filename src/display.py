# -*- coding: utf-8 -*-
"""
Display module for defect inspection system.
Handles visualization of detection results, system status, and statistics.
"""

import os
import cv2
import time
import numpy as np
import threading
import logging
from typing import List, Dict, Tuple, Optional, Union

# Set up logging
logger = logging.getLogger("DefectInspection.Display")

class DisplayManager:
    """Manages the visualization window and UI elements"""
    
    def __init__(self, window_name: str = "Defect Inspection", 
                 fullscreen: bool = False,
                 resolution: Tuple[int, int] = (1920, 1080),
                 font_scale: float = 0.7):
        """
        Initialize display manager
        
        Args:
            window_name: Name of display window
            fullscreen: Whether to use fullscreen mode
            resolution: Display resolution (width, height)
            font_scale: Base font scale for text display
        """
        self.window_name = window_name
        self.fullscreen = fullscreen
        self.width, self.height = resolution
        self.font_scale = font_scale
        
        # Initialize stats display
        self.stats = {
            "total_inspections": 0,
            "defects_found": 0,
            "defect_rate": 0.0,
            "average_processing_time": 0.0,
            "system_uptime": 0
        }
        
        # Initialize status
        self.status_text = ["System ready", "Waiting for product"]
        self.status_color = (0, 255, 0)  # Green by default
        
        # Initialize image buffers
        self.current_image = self._create_default_image()
        self.display_buffer = self._create_default_image()
        
        # Create window
        if fullscreen:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.width, self.height)
        
        # Start display thread
        self.running = True
        self.lock = threading.Lock()
        self.display_thread = threading.Thread(target=self._display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()
        
        logger.info(f"Display initialized ({resolution[0]}x{resolution[1]}, {'fullscreen' if fullscreen else 'windowed'})")
    
    def _create_default_image(self) -> np.ndarray:
        """Create default display image"""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add title and instructions
        title = "Defect Inspection System"
        cv2.putText(img, title, (self.width//2 - 200, self.height//2 - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                   
        text = "Waiting for camera input..."
        cv2.putText(img, text, (self.width//2 - 150, self.height//2 + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 1)
        
        return img
    
    def _display_loop(self):
        """Main display loop that runs in separate thread"""
        last_refresh = time.time()
        while self.running:
            # Limit refresh rate to ~30 FPS
            if time.time() - last_refresh < 0.033:
                time.sleep(0.005)
                continue
                
            last_refresh = time.time()
            
            # Create a copy of the current display buffer to avoid race conditions
            with self.lock:
                display_copy = self.display_buffer.copy()
            
            # Show the image
            cv2.imshow(self.window_name, display_copy)
            
            # Process key events (1ms wait)
            self.process_keys()
    
    def update_image(self, image: np.ndarray):
        """
        Update the display with a new image
        
        Args:
            image: New image to display
        """
        if image is None:
            return
            
        # Resize image to fit display if needed
        h, w = image.shape[:2]
        if w != self.width or h != self.height:
            image = cv2.resize(image, (self.width, self.height))
        
        with self.lock:
            self.current_image = image.copy()
            self.display_buffer = image.copy()
    
    def show_image_with_overlay(self, image: np.ndarray, status_text: List[str] = None, 
                               status_color: Tuple[int, int, int] = None):
        """
        Show image with status overlay
        
        Args:
            image: Image to display
            status_text: List of status text lines
            status_color: Color for status text (B,G,R)
        """
        if image is None:
            return
            
        # Update status
        if status_text:
            self.status_text = status_text
        
        if status_color:
            self.status_color = status_color
            
        # Resize image if needed
        h, w = image.shape[:2]
        if w != self.width or h != self.height:
            display_img = cv2.resize(image, (self.width, self.height))
        else:
            display_img = image.copy()
            
        # Add status overlay
        self._add_status_overlay(display_img)
        
        # Add statistics
        self._add_stats_overlay(display_img)
        
        with self.lock:
            self.current_image = image.copy()  # Store original
            self.display_buffer = display_img  # Display with overlays
    
    def show_split_view(self, original_image: np.ndarray, processed_image: np.ndarray,
                       status_text: List[str] = None, status_color: Tuple[int, int, int] = None):
        """
        Show split view with original and processed images side by side
        
        Args:
            original_image: Original camera image
            processed_image: Processed image with detections
            status_text: List of status text lines
            status_color: Color for status text
        """
        if original_image is None or processed_image is None:
            return
            
        # Update status
        if status_text:
            self.status_text = status_text
        
        if status_color:
            self.status_color = status_color
        
        # Create a combined image for side-by-side view
        combined_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Calculate dimensions for each panel
        panel_width = self.width // 2
        
        # Resize both images to fit in panels
        orig_resized = cv2.resize(original_image, (panel_width, self.height))
        proc_resized = cv2.resize(processed_image, (panel_width, self.height))
        
        # Place images side by side
        combined_img[:, 0:panel_width] = orig_resized
        combined_img[:, panel_width:] = proc_resized
        
        # Add a dividing line
        cv2.line(combined_img, (panel_width, 0), (panel_width, self.height), (200, 200, 200), 2)
        
        # Add labels
        cv2.putText(combined_img, "Original", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined_img, "Processed", (panel_width + 20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add status overlay
        self._add_status_overlay(combined_img)
        
        # Add statistics
        self._add_stats_overlay(combined_img)
        
        with self.lock:
            self.display_buffer = combined_img
    
    def _add_status_overlay(self, image: np.ndarray):
        """Add status overlay to image"""
        if not self.status_text:
            return
            
        # Create dark overlay at the bottom of the image
        overlay_height = 40 * len(self.status_text) + 20
        overlay = image[-overlay_height:].copy()
        cv2.rectangle(overlay, (0, 0), (self.width, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image[-overlay_height:], 0.3, 0, image[-overlay_height:])
        
        # Add status text
        for i, text in enumerate(self.status_text):
            y_pos = image.shape[0] - overlay_height + 30 + i * 40
            cv2.putText(image, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_DUPLEX, self.font_scale, self.status_color, 1)
    
    def _add_stats_overlay(self, image: np.ndarray):
        """Add statistics overlay to image"""
        # Create transparent overlay in top-right corner
        overlay_width = 350
        overlay_height = 130
        x_start = self.width - overlay_width - 10
        y_start = 10
        
        overlay = image[y_start:y_start + overlay_height, x_start:x_start + overlay_width].copy()
        cv2.rectangle(overlay, (0, 0), (overlay_width, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, 
                        image[y_start:y_start + overlay_height, x_start:x_start + overlay_width], 
                        0.3, 0, image[y_start:y_start + overlay_height, x_start:x_start + overlay_width])
        
        # Add stats text
        font_scale = self.font_scale - 0.1
        cv2.putText(image, "System Statistics", (x_start + 10, y_start + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
                   
        cv2.putText(image, f"Inspections: {self.stats['total_inspections']}", 
                   (x_start + 10, y_start + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 220, 220), 1)
                   
        cv2.putText(image, f"Defects: {self.stats['defects_found']} ({self.stats['defect_rate']:.1f}%)", 
                   (x_start + 10, y_start + 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 220, 220), 1)
                   
        hours = self.stats['system_uptime'] // 3600
        minutes = (self.stats['system_uptime'] % 3600) // 60
        seconds = self.stats['system_uptime'] % 60
        cv2.putText(image, f"Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}", 
                   (x_start + 10, y_start + 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 220, 220), 1)
    
    def update_stats(self, stats: Dict):
        """
        Update statistics
        
        Args:
            stats: Dictionary with updated statistics
        """
        self.stats.update(stats)
    
    def process_keys(self):
        """
        Process keyboard input
        
        Returns:
            int: Key code of pressed key or -1 if no key pressed
        """
        key = cv2.waitKey(1) & 0xFF
        
        # ESC key
        if key == 27:
            logger.info("ESC key pressed, initiating shutdown")
            return key
            
        # F key for fullscreen toggle
        elif key == ord('f'):
            self.toggle_fullscreen()
            
        # Any other key press
        elif key != 255:
            return key
            
        return -1
    
    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode"""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            logger.debug("Switched to fullscreen mode")
        else:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.width, self.height)
            logger.debug("Switched to windowed mode")
    
    def close(self):
        """Close display and cleanup"""
        self.running = False
        # Wait for display thread to finish
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=1.0)
        
        # Destroy window
        cv2.destroyWindow(self.window_name)
        logger.info("Display closed")


# For testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create display manager
    display = DisplayManager(fullscreen=False, resolution=(1280, 720))
    
    # Test with some sample images and data
    try:
        # Create a test image
        test_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (100, 100), (400, 400), (0, 255, 0), 2)
        cv2.circle(test_img, (300, 500), 50, (0, 0, 255), -1)
        
        # Show status and stats
        display.update_stats({
            "total_inspections": 125,
            "defects_found": 23,
            "defect_rate": 18.4,
            "average_processing_time": 45.2,
            "system_uptime": 3665  # 1 hour, 1 minute, 5 seconds
        })
        
        # Create detection-style image
        processed_img = test_img.copy()
        cv2.rectangle(processed_img, (100, 100), (400, 400), (255, 0, 0), 2)
        cv2.putText(processed_img, "Defect: scratch (0.87)", (105, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Show split view
        display.show_split_view(
            test_img, 
            processed_img, 
            ["Product ID: PROD_12345", "Status: Defect detected", "Confidence: 0.87"]
        )
        
        # Wait for user to press ESC
        logger.info("Press ESC to exit test")
        while True:
            key = display.process_keys()
            if key == 27:  # ESC
                break
            time.sleep(0.1)
            
    finally:
        # Clean up
        display.close()
        logger.info("Test completed")