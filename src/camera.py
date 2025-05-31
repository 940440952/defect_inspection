# -*- coding: utf-8 -*-
# Optimized camera module for industrial camera control:
import os
import sys
import time
import logging
import numpy as np
from datetime import datetime
from ctypes import *

# Add path for camera SDK
sys.path.append("/home/gtm/defect_inspection/dependencies/MvImport")
from MvCameraControl_class import *

logger = logging.getLogger("DefectInspection.Camera")

class CameraManager:
    """Camera management class that keeps the connection open between captures"""
    
    def __init__(self, device_index=0):
        self.device_index = device_index
        self.cam = None
        self.is_initialized = False
        self.is_open = False
        self.stDeviceInfo = None
        self.packet_size_set = False
        
        # Initialize SDK once
        if MvCamera.MV_CC_Initialize() != 0:
            raise RuntimeError("Failed to initialize camera SDK")
        self.is_initialized = True
        logger.info("Camera SDK initialized")

    def open_camera(self):
        """Open and configure the camera for capture"""
        if self.is_open:
            return True
            
        start_time = time.time()
        
        # Enumerate devices
        deviceList = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
        if ret != 0 or deviceList.nDeviceNum == 0:
            logger.error("No cameras detected")
            return False
            
        if self.device_index >= deviceList.nDeviceNum:
            logger.error(f"Camera index {self.device_index} out of range, only {deviceList.nDeviceNum} available")
            return False
            
        logger.info(f"Using camera {self.device_index} of {deviceList.nDeviceNum}")
        
        # Create camera instance
        self.cam = MvCamera()
        self.stDeviceInfo = cast(deviceList.pDeviceInfo[self.device_index], 
                               POINTER(MV_CC_DEVICE_INFO)).contents
        
        # Create handle
        if self.cam.MV_CC_CreateHandle(self.stDeviceInfo) != 0:
            logger.error("Failed to create camera handle")
            return False
            
        # Open device with exclusive access
        if self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0:
            logger.error("Failed to open camera")
            self.cam.MV_CC_DestroyHandle()
            self.cam = None
            return False
            
        # Set trigger mode off for continuous capture
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            logger.error(f"Failed to set trigger mode (0x{ret:x})")
            return False

        # Set packet size for GigE cameras (only once to avoid the 1.4s delay)
        if self.stDeviceInfo.nTLayerType == MV_GIGE_DEVICE and not self.packet_size_set:
            # Use a fixed packet size instead of querying optimal size each time
            ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", 1500)  # Standard MTU size
            if ret != 0:
                logger.warning(f"Set packet size failed (0x{ret:x})")
            self.packet_size_set = True
        
        self.is_open = True
        logger.info(f"Camera opened and configured in {(time.time() - start_time)*1000:.2f}ms")
        return True

    def capture_image(self, return_array=True, save_path=None):
        """
        Capture an image from the camera
        
        Args:
            return_array (bool): Whether to return the image as numpy array
            save_path (str): Path to save image (if None, image is not saved)
            
        Returns:
            tuple: (image_array, image_path) or (None, None) if failed
        """
        if not self.is_open and not self.open_camera():
            logger.error("Failed to open camera for capture")
            return None, None
            
        start_time = time.time()
        
        try:
            # Start grabbing
            if self.cam.MV_CC_StartGrabbing() != 0:
                logger.error("Failed to start grabbing")
                return None, None
                
            # Get one frame
            stFrame = MV_FRAME_OUT()
            memset(byref(stFrame), 0, sizeof(stFrame))
            ret = self.cam.MV_CC_GetImageBuffer(stFrame, 1000)
            if ret != 0:
                logger.error(f"Failed to get image (0x{ret:x})")
                self.cam.MV_CC_StopGrabbing()
                return None, None
            
            image_path = None
            image_array = None
            
            if return_array or save_path:
                # Process image based on pixel format - using direct memory access to avoid copying
                if stFrame.stFrameInfo.enPixelType == PixelType_Gvsp_Mono8:
                    # For monochrome images
                    data = np.frombuffer(
                        (c_ubyte * stFrame.stFrameInfo.nFrameLen).from_address(
                            addressof(stFrame.pBufAddr.contents)),
                        dtype=np.uint8,
                        count=int(stFrame.stFrameInfo.nWidth * stFrame.stFrameInfo.nHeight)
                    )
                    image_array = data.reshape(stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth)
                
                elif stFrame.stFrameInfo.enPixelType == PixelType_Gvsp_RGB8_Packed:
                    # For RGB images - direct access to avoid copying
                    data = np.frombuffer(
                        (c_ubyte * stFrame.stFrameInfo.nFrameLen).from_address(
                            addressof(stFrame.pBufAddr.contents)),
                        dtype=np.uint8,
                        count=int(stFrame.stFrameInfo.nWidth * stFrame.stFrameInfo.nHeight * 3)
                    )
                    
                    # Reshape and convert RGB to BGR for OpenCV
                    image_array = data.reshape(stFrame.stFrameInfo.nHeight, 
                                              stFrame.stFrameInfo.nWidth, 
                                              3)
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    
                else:
                    # For other formats, use the SDK's conversion to BGR
                    nBGRSize = stFrame.stFrameInfo.nWidth * stFrame.stFrameInfo.nHeight * 3
                    stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
                    memset(byref(stConvertParam), 0, sizeof(stConvertParam))
                    stConvertParam.nWidth = stFrame.stFrameInfo.nWidth
                    stConvertParam.nHeight = stFrame.stFrameInfo.nHeight
                    stConvertParam.pSrcData = stFrame.pBufAddr
                    stConvertParam.nSrcDataLen = stFrame.stFrameInfo.nFrameLen
                    stConvertParam.enSrcPixelType = stFrame.stFrameInfo.enPixelType
                    stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed
                    stConvertParam.pDstBuffer = (c_ubyte * nBGRSize)()
                    stConvertParam.nDstBufferSize = nBGRSize
                    
                    ret = self.cam.MV_CC_ConvertPixelType(stConvertParam)
                    if ret != 0:
                        logger.error(f"Failed to convert pixel format (0x{ret:x})")
                        self.cam.MV_CC_FreeImageBuffer(stFrame)
                        self.cam.MV_CC_StopGrabbing()
                        return None, None
                        
                    # Create numpy array from converted data
                    image_array = np.frombuffer(stConvertParam.pDstBuffer, 
                                               dtype=np.uint8, 
                                               count=nBGRSize).reshape(
                                               stFrame.stFrameInfo.nHeight, 
                                               stFrame.stFrameInfo.nWidth, 3)
                
                # Save the image if a path is provided
                if save_path and image_array is not None:
                    import cv2
                    if cv2.imwrite(save_path, image_array, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                        image_path = save_path
                        logger.debug(f"Saved image to {save_path}")
                    else:
                        logger.error(f"Failed to save image to {save_path}")
            
            # Free image buffer and stop grabbing
            self.cam.MV_CC_FreeImageBuffer(stFrame)
            self.cam.MV_CC_StopGrabbing()
            
            logger.debug(f"Image capture completed in {(time.time() - start_time)*1000:.2f}ms")
            return image_array, image_path
            
        except Exception as e:
            logger.exception(f"Error capturing image: {e}")
            try:
                self.cam.MV_CC_StopGrabbing()
            except:
                pass
            return None, None
    
    def close_camera(self):
        """Close the camera connection"""
        if self.cam and self.is_open:
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.is_open = False
            logger.info("Camera closed")

    def __del__(self):
        """Clean up resources when object is destroyed"""
        self.close_camera()
        if self.is_initialized:
            MvCamera.MV_CC_Finalize()

# test
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    camera = CameraManager(device_index=0)
    if camera.open_camera():
        image, path = camera.capture_image(return_array=True, save_path="test_image.jpg")
        # 查看image是什么类型的
        if isinstance(image, np.ndarray):
            print(f"Captured image type: {type(image)}")
        else:
            print(f"Captured image type: {type(image)}")
        # 查看image的shape 
        if image is not None:
            print(f"Captured image shape: {image.shape}")
        camera.close_camera()
    else:
        print("Failed to open camera")