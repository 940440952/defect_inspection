# -*- coding: utf-8 -*-
"""
Conveyor belt control module for defect inspection system
Handles GPIO interactions for motor control and sensor inputs
"""

import time
import logging
import threading
from enum import Enum

try:
    import RPi.GPIO as GPIO
except ImportError:
    # For development/testing on non-Pi platforms
    import mock_gpio as GPIO
    logging.warning("Using mock GPIO - conveyor control simulated")

# Set up logging
logger = logging.getLogger("DefectInspection.Conveyor")

class ConveyorState(Enum):
    """Enum for conveyor state tracking"""
    STOPPED = 0
    RUNNING = 1
    ERROR = 2


class ConveyorController:
    """Controls the conveyor belt movement via GPIO"""
    
    def __init__(self, enable_pin=17, direction_pin=27, pulse_pin=22, sensor_pin=23,
                 emergency_stop_pin=24, default_speed=50):
        """
        Initialize conveyor controller with GPIO pins
        
        Args:
            enable_pin: GPIO pin for motor driver enable
            direction_pin: GPIO pin for motor direction control
            pulse_pin: GPIO pin for motor speed control (PWM)
            sensor_pin: GPIO pin for product detection sensor
            emergency_stop_pin: GPIO pin for emergency stop button
            default_speed: Default speed percentage (0-100)
        """
        # Configuration
        self.enable_pin = enable_pin
        self.direction_pin = direction_pin
        self.pulse_pin = pulse_pin
        self.sensor_pin = sensor_pin
        self.emergency_stop_pin = emergency_stop_pin
        self.default_speed = default_speed
        self.pwm_frequency = 100  # PWM frequency in Hz
        
        # State tracking
        self.state = ConveyorState.STOPPED
        self.pwm = None
        self.last_sensor_time = 0
        self.product_detected = False
        self.emergency_stop_pressed = False
        self._lock = threading.Lock()  # For thread-safe operations
        
        # Initialize GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.enable_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.direction_pin, GPIO.OUT, initial=GPIO.HIGH)  # Forward direction
        GPIO.setup(self.pulse_pin, GPIO.OUT, initial=GPIO.LOW)
        
        # Configure sensor as input with pull-up (normally closed sensor)
        GPIO.setup(self.sensor_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(self.sensor_pin, GPIO.BOTH, callback=self._sensor_callback, bouncetime=100)
        
        # Configure emergency stop button with pull-up (normally open)
        GPIO.setup(self.emergency_stop_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(self.emergency_stop_pin, GPIO.FALLING, 
                             callback=self._emergency_stop_callback, bouncetime=300)
        
        # Configure PWM
        self.pwm = GPIO.PWM(self.pulse_pin, self.pwm_frequency)
        self.pwm.start(0)  # Start with 0% duty cycle (stopped)
        
        # Ensure conveyor is stopped initially
        self.stop()
        logger.info("Conveyor controller initialized")
    
    def _sensor_callback(self, channel):
        """Callback for product detection sensor state change"""
        if GPIO.input(self.sensor_pin) == 0:  # Product detected (sensor triggered)
            self.product_detected = True
            self.last_sensor_time = time.time()
            logger.debug("Product detected by sensor")
        else:  # Product no longer detected
            self.product_detected = False
            logger.debug("Product cleared sensor")
    
    def _emergency_stop_callback(self, channel):
        """Callback for emergency stop button press"""
        if GPIO.input(self.emergency_stop_pin) == 0:  # Button pressed (active low)
            self.emergency_stop_pressed = True
            logger.warning("Emergency stop activated")
            self.stop()  # Stop the conveyor immediately
    
    def start(self, speed=None):
        """
        Start the conveyor belt at specified speed
        
        Args:
            speed: Speed percentage (0-100), uses default if None
        
        Returns:
            bool: True if started successfully, False if emergency stop active
        """
        with self._lock:
            if self.emergency_stop_pressed:
                logger.warning("Cannot start conveyor: Emergency stop active")
                return False
            
            if speed is None:
                speed = self.default_speed
            
            # Clamp speed to valid range
            speed = max(0, min(100, speed))
            
            # Activate motor driver
            GPIO.output(self.enable_pin, GPIO.HIGH)
            GPIO.output(self.direction_pin, GPIO.HIGH)  # Forward direction
            self.pwm.ChangeDutyCycle(speed)
            self.state = ConveyorState.RUNNING
            
            logger.debug(f"Conveyor started at speed {speed}%")
            return True
    
    def stop(self):
        """
        Stop the conveyor belt
        
        Returns:
            bool: True if stopped successfully
        """
        with self._lock:
            self.pwm.ChangeDutyCycle(0)
            GPIO.output(self.enable_pin, GPIO.LOW)
            self.state = ConveyorState.STOPPED
            logger.debug("Conveyor stopped")
            return True
    
    def is_product_detected(self):
        """
        Check if product is detected by sensor
        
        Returns:
            bool: True if product detected
        """
        return self.product_detected
    
    def is_running(self):
        """
        Check if conveyor is currently running
        
        Returns:
            bool: True if running
        """
        return self.state == ConveyorState.RUNNING
    
    def wait_for_product(self, timeout=None):
        """
        Wait until a product is detected by the sensor
        
        Args:
            timeout: Maximum time to wait in seconds, None for no timeout
            
        Returns:
            bool: True if product detected, False if timeout
        """
        start_time = time.time()
        while not self.is_product_detected():
            # Check timeout
            if timeout and time.time() - start_time > timeout:
                logger.debug(f"Timeout waiting for product detection ({timeout}s)")
                return False
            time.sleep(0.01)  # Small sleep to avoid CPU hogging
        
        return True
    
    def reset_emergency_stop(self):
        """
        Reset emergency stop state after the button has been physically reset
        
        Returns:
            bool: True if reset successfully
        """
        if GPIO.input(self.emergency_stop_pin) != 0:  # Button is no longer pressed
            self.emergency_stop_pressed = False
            logger.info("Emergency stop reset")
            return True
        else:
            logger.warning("Cannot reset emergency stop: Button still active")
            return False
    
    def get_state(self):
        """Get current conveyor state"""
        if self.emergency_stop_pressed:
            return ConveyorState.ERROR
        return self.state
    
    def cleanup(self):
        """Clean up GPIO resources"""
        if self.pwm:
            self.pwm.stop()
        
        # Remove event detection callbacks
        try:
            GPIO.remove_event_detect(self.sensor_pin)
            GPIO.remove_event_detect(self.emergency_stop_pin)
        except:
            pass
        
        # Clean up only our used pins rather than all GPIO
        pins = [self.enable_pin, self.direction_pin, self.pulse_pin, 
                self.sensor_pin, self.emergency_stop_pin]
        for pin in pins:
            try:
                GPIO.cleanup(pin)
            except:
                pass
        
        logger.info("Conveyor controller resources cleaned up")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.stop()
            self.cleanup()
        except:
            pass


# For testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create controller with default pins
    conveyor = ConveyorController()
    
    try:
        print("Starting conveyor for 3 seconds...")
        conveyor.start(speed=40)
        time.sleep(3)
        
        print("Stopping conveyor...")
        conveyor.stop()
        time.sleep(1)
        
        print("Waiting for product detection (5 second timeout)...")
        if conveyor.wait_for_product(timeout=5):
            print("Product detected!")
        else:
            print("No product detected within timeout.")
        
    except KeyboardInterrupt:
        print("Test interrupted")
    finally:
        conveyor.stop()
        conveyor.cleanup()
        print("Test completed")