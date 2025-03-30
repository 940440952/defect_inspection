#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import signal
import logging
import argparse
import threading
from datetime import datetime

# Import our modules
from src.camera import CameraManager
from src.conveyor import ConveyorController
from src.detector import YOLODetector
from src.display import DisplayManager
from src.api_client import APIClient
from src.utils import setup_logging, load_config

# Global control flags
running = True
stats = {
    "total_inspections": 0, # Total inspections performed
    "defects_found": 0, # Total defects found
    "ejections": 0, # Total number of defect ejections performed by the system
    "start_time": time.time()
}

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print("\nShutting down...")
    running = False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Defect Inspection System')
    parser.add_argument('--config', type=str, default='config/settings.json',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, help='Override model path')
    parser.add_argument('--api-url', type=str, help='Override API URL')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging') # Debug flag
    parser.add_argument('--no-display', action='store_true', help='Disable display') # Disable display flag
    parser.add_argument('--no-ejection', action='store_true', help='Disable defect ejection')
    parser.add_argument('--test-ejector', action='store_true', help='Test ejector functionality')
    return parser.parse_args()

def display_statistics(logger):
    """Display system statistics"""
    global stats
    
    runtime = time.time() - stats["start_time"]
    hours = int(runtime / 3600)
    minutes = int((runtime % 3600) / 60)
    seconds = int(runtime % 60)
    
    if stats["total_inspections"] > 0:
        defect_rate = (stats["defects_found"] / stats["total_inspections"]) * 100
        ejection_rate = (stats["ejections"] / stats["total_inspections"]) * 100
    else:
        defect_rate = 0
        ejection_rate = 0
    
    logger.info(f"System Statistics:")
    logger.info(f"  Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")
    logger.info(f"  Inspections: {stats['total_inspections']}")
    logger.info(f"  Defects found: {stats['defects_found']} ({defect_rate:.1f}%)")
    logger.info(f"  Ejections: {stats['ejections']} ({ejection_rate:.1f}%)")
    
    if stats["total_inspections"] > 0:
        throughput = stats["total_inspections"] / (runtime / 3600)
        logger.info(f"  Throughput: {throughput:.1f} units/hour")

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(log_level)
    logger.info("Starting Defect Inspection System")
    
    # Override config with command line arguments
    if args.model:
        config['model_path'] = args.model
    if args.api_url:
        config['api_url'] = args.api_url
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Initialize components
        logger.info("Initializing system components...")
        
        # Initialize camera
        camera = CameraManager(config.get('camera_index', 0))
        
        # Initialize conveyor controller with ejector pin
        conveyor = ConveyorController(
            enable_pin=config.get('enable_pin', 17),
            direction_pin=config.get('direction_pin', 27),
            pulse_pin=config.get('pulse_pin', 22),
            sensor_pin=config.get('sensor_pin', 23),
            emergency_stop_pin=config.get('emergency_stop_pin', 24)
        )
        
        # Add ejector control
        ejector_pin = config.get('ejector_pin', 25)
        ejector_active_high = config.get('ejector_active_high', True)
        
        # Set up ejector pin
        try:
            import RPi.GPIO as GPIO
            GPIO.setup(ejector_pin, GPIO.OUT, initial=GPIO.LOW if ejector_active_high else GPIO.HIGH)
            logger.info(f"Ejector configured on pin {ejector_pin} (active {'high' if ejector_active_high else 'low'})")
            
            # Add ejector method to conveyor controller
            def eject_defect(delay=0, duration=0.5):
                """Activate ejector after delay"""
                def _eject():
                    time.sleep(delay)
                    logger.debug(f"Activating ejector for {duration}s")
                    GPIO.output(ejector_pin, GPIO.HIGH if ejector_active_high else GPIO.LOW)
                    time.sleep(duration)
                    GPIO.output(ejector_pin, GPIO.LOW if ejector_active_high else GPIO.HIGH)
                    logger.debug("Ejector deactivated")
                    stats["ejections"] += 1
                
                # Run ejector in separate thread to avoid blocking
                thread = threading.Thread(target=_eject)
                thread.daemon = True
                thread.start()
                return True
            
            # Attach method to conveyor controller
            conveyor.eject_defect = eject_defect
            
        except (ImportError, NameError):
            # Mock ejector for development environments
            logger.warning("GPIO not available, using simulated ejector")
            def mock_eject_defect(delay=0, duration=0.5):
                def _mock_eject():
                    time.sleep(delay)
                    logger.info(f"[SIMULATION] Ejector activated for {duration}s")
                    time.sleep(duration)
                    logger.info("[SIMULATION] Ejector deactivated")
                    stats["ejections"] += 1
                
                thread = threading.Thread(target=_mock_eject)
                thread.daemon = True
                thread.start()
                return True
            
            conveyor.eject_defect = mock_eject_defect
        
        # Test ejector if requested
        if args.test_ejector:
            logger.info("Testing ejector functionality...")
            conveyor.eject_defect(delay=0.5, duration=1.0)
            time.sleep(2)  # Wait for ejector test to complete
        
        # Initialize YOLO detector
        detector = YOLODetector(
            model_path=config.get('model_path', 'models/defect_model.onnx'),
            conf_thresh=config.get('confidence_threshold', 0.25)
        )
        
        # Initialize display if not disabled
        display = None
        if not args.no_display:
            try:
                display = DisplayManager(
                    window_name=config.get('window_name', 'Defect Inspection'),
                    fullscreen=config.get('fullscreen', False),
                    resolution=config.get('display_resolution', (1920, 1080))
                )
                logger.info("Display initialized")
            except Exception as e:
                logger.error(f"Failed to initialize display: {e}")
                display = None
        
        # Initialize API client
        api_client = None
        if config.get('api_url'):
            api_client = APIClient(
                api_url=config.get('api_url'),
                auth_token=config.get('auth_token', '')
            )
            logger.info(f"API client initialized with endpoint: {config.get('api_url')}")
        
        logger.info("System initialized, starting inspection loop")
        
        # Create output directory if it doesn't exist
        output_dir = config.get('output_dir', 'data/images')
        os.makedirs(output_dir, exist_ok=True)
        
        # Start conveyor
        conveyor.start(config.get('conveyor_speed', 50))
        
        # Ejection settings
        ejection_enabled = not args.no_ejection and config.get('ejection_enabled', True)
        ejection_threshold = config.get('ejection_threshold', 0.5)
        ejection_delay = config.get('ejection_delay', 1.0)  # Delay before ejection in seconds
        ejection_duration = config.get('ejection_duration', 0.5)  # Duration of ejection signal
        
        if ejection_enabled:
            logger.info(f"Defect ejection enabled (threshold: {ejection_threshold}, delay: {ejection_delay}s)")
        else:
            logger.info("Defect ejection disabled")
        
        # Statistics timer
        last_stats_time = time.time()
        stats_interval = config.get('stats_interval', 3600)  # Default: show stats every hour
        
        # Main inspection loop
        while running:
            # Display periodic statistics
            if time.time() - last_stats_time > stats_interval:
                display_statistics(logger)
                last_stats_time = time.time()
            
            # Check if a product is detected
            if conveyor.is_product_detected():
                logger.info("Product detected, stopping conveyor")
                
                # Stop the conveyor
                conveyor.stop()
                
                # Wait for conveyor to fully stop and product to stabilize
                time.sleep(config.get('stop_delay', 0.5))
                
                # Capture image
                logger.info("Capturing image")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"{output_dir}/capture_{timestamp}.jpg"
                image, saved_path = camera.capture_image(save_path=image_filename)
                
                if image is None:
                    logger.error("Failed to capture image")
                    conveyor.start(config.get('conveyor_speed', 50))
                    continue
                
                # Run detection
                logger.info("Running defect detection")
                detection_start = time.time()
                detections = detector.detect(image)
                detection_time = (time.time() - detection_start) * 1000
                logger.info(f"Detection completed in {detection_time:.2f}ms, found {len(detections)} defects")
                
                # Update statistics
                stats["total_inspections"] += 1
                if detections:
                    stats["defects_found"] += 1
                
                # Create metadata
                metadata = {
                    'timestamp': timestamp,
                    'detection_time_ms': detection_time,
                    'product_id': f"PROD_{timestamp}",
                    'system_info': {
                        'device': 'Jetson Orin NX',
                        'software_version': config.get('version', '1.0')
                    },
                    'has_defects': len(detections) > 0,
                    'defects_count': len(detections)
                }
                
                # Add defect details to metadata
                if detections:
                    defect_details = []
                    for i, det in enumerate(detections):
                        x1, y1, x2, y2, conf, class_id = det
                        class_name = detector.class_names[int(class_id)] if class_id < len(detector.class_names) else f"unknown_{class_id}"
                        defect_details.append({
                            'id': i,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class_id': int(class_id),
                            'class_name': class_name
                        })
                    metadata['defects'] = defect_details
                
                # Upload results to API in a separate thread to avoid blocking
                if api_client and saved_path:
                    upload_thread = threading.Thread(
                        target=api_client.upload_result,
                        args=(saved_path, detections, metadata)
                    )
                    upload_thread.daemon = True
                    upload_thread.start()
                    logger.debug(f"Started upload thread for image {os.path.basename(saved_path)}")
                
                # Display results
                if display and image is not None:
                    # Draw defects on the image
                    display_image = detector.draw_detections(image, detections) if detections else image.copy()
                    
                    # Add status overlay
                    status_text = []
                    status_text.append(f"Product ID: PROD_{timestamp}")
                    status_text.append(f"Defects: {len(detections)}")
                    if len(detections) > 0:
                        max_conf = max([det[4] for det in detections])
                        status_text.append(f"Max confidence: {max_conf:.2f}")
                        status_text.append(f"Ejection: {'YES' if max_conf >= ejection_threshold else 'NO'}")
                    
                    display.show_image_with_overlay(display_image, status_text)
                    
                    # Wait for display time (if configured)
                    display_time = config.get('display_time', 2.0)
                    if display_time > 0:
                        start_wait = time.time()
                        while time.time() - start_wait < display_time and running:
                            key = display.process_keys()
                            if key == 27:  # ESC key
                                running = False
                                break
                            time.sleep(0.01)
                
                # Handle defect ejection
                if ejection_enabled and detections:
                    # Check if any defect has high enough confidence
                    should_eject = False
                    max_confidence = 0.0
                    
                    for det in detections:
                        confidence = det[4] if len(det) > 4 else 0.0
                        max_confidence = max(max_confidence, confidence)
                        if confidence >= ejection_threshold:
                            should_eject = True
                            break
                    
                    if should_eject:
                        logger.info(f"Defect detected with confidence {max_confidence:.2f}, scheduling ejection")
                        # Schedule ejection after delay
                        conveyor.eject_defect(delay=ejection_delay, duration=ejection_duration)
                    else:
                        logger.info(f"Defect confidence {max_confidence:.2f} below threshold, not ejecting")
                
                # Restart conveyor
                logger.info("Restarting conveyor")
                conveyor.start(config.get('conveyor_speed', 50))
            
            # Small delay to prevent CPU overuse
            time.sleep(0.01)
            
    except Exception as e:
        logger.exception(f"Error in main loop: {e}")
        
    finally:
        # Display final statistics
        display_statistics(logger)
        
        # Cleanup resources
        logger.info("Cleaning up resources...")
        
        try:
            if 'display' in locals() and display:
                display.close()
        except Exception as e:
            logger.error(f"Error closing display: {e}")
        
        try:
            if 'conveyor' in locals():
                conveyor.stop()
                conveyor.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up conveyor: {e}")
        
        try:
            if 'camera' in locals():
                camera.close_camera()
        except Exception as e:
            logger.error(f"Error closing camera: {e}")
        
        logger.info("System shutdown complete")

if __name__ == "__main__":
    main()