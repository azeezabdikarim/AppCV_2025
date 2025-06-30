#!/usr/bin/env python3

"""
Line Following Module - Week 1 Assignment
=========================================

STUDENT IMPLEMENTATION FOCUS:
Students implement these core computer vision functions:
1. _extract_roi() - Manually crop image region
2. _convert_to_grayscale() - Convert BGR to grayscale
3. _detect_line_center() - Find the center of line in image
4. _calculate_error() - Compute steering error

The debug system works immediately with dummy values, then improves as 
students implement each function properly.
"""

import cv2
import numpy as np
import time

class LineFollower:
    def __init__(self):
        """Initialize the line follower"""
        
        # =================================================================
        # PID CONTROLLER (PROVIDED - Students focus on computer vision)
        # =================================================================
        self.Kp = 0.8    # Proportional gain
        self.Ki = 0.1    # Integral gain  
        self.Kd = 0.3    # Derivative gain
        
        # PID state variables
        self.previous_error = 0.0
        self.integral_error = 0.0
        self.last_time = time.time()
        
        # =================================================================
        # STUDENT TUNABLE PARAMETERS
        # Students can adjust these values to improve performance
        # =================================================================
        self.roi_top = 0.6        # Start ROI at 60% down the image
        self.roi_bottom = 0.9     # End ROI at 90% down the image
        self.roi_left = 0.1       # Start ROI at 10% from left
        self.roi_right = 0.9      # End ROI at 90% from left
        
        self.canny_low = 50       # Lower threshold for edge detection
        self.canny_high = 150     # Upper threshold for edge detection
        
        # =================================================================
        # STUDENT IMPLEMENTATION VARIABLES
        # These start as dummy values - students implement the functions that set them
        # =================================================================
        
        # Region of Interest - students implement _extract_roi()
        self.current_roi = None           # Will hold cropped image
        self.roi_bounds = (0, 0, 0, 0)   # (top, bottom, left, right) pixel coordinates
        
        # Grayscale conversion - students implement _convert_to_grayscale()  
        self.gray_image = None            # Will hold grayscale version
        
        # Line detection - students implement _detect_line_center()
        self.line_center_x = None         # X coordinate of detected line center
        self.line_center_y = None         # Y coordinate of detected line center
        
        # Error calculation - students implement _calculate_error()
        self.current_error = 0.0          # Pixel error (image_center - line_center)
        self.image_center_x = None        # Center X coordinate of ROI
        
        # =================================================================
        # DEBUG SYSTEM (PROVIDED - Works with dummy values)
        # =================================================================
        self.debug_frame = None
        self.current_debug_data = {
            'error_px': 0.0,
            'steering_angle': 0.0,
            'lines_detected': 0,
            'implementation_status': {
                'roi_extraction': False,
                'grayscale_conversion': False,
                'line_detection': False,
                'error_calculation': False
            }
        }
        
        print("✅ Line follower initialized")
        print("🎯 TODO: Implement the 4 core computer vision functions")
    
    def compute_steering_angle(self, frame, debug_level=0):
        """
        Main processing pipeline - calls student-implemented functions
        """
        if frame is None:
            return 0.0
        
        try:
            # Step 1: Extract Region of Interest (STUDENT IMPLEMENTS)
            self._extract_roi(frame)
            
            # Step 2: Convert to grayscale (STUDENT IMPLEMENTS)
            self._convert_to_grayscale()
            
            # Step 3: Detect line center (STUDENT IMPLEMENTS)
            self._detect_line_center()
            
            # Step 4: Calculate error (STUDENT IMPLEMENTS)
            self._calculate_error()
            
            # Step 5: Apply PID control (PROVIDED)
            steering_angle = self._apply_pid_control()
            
            # Step 6: Update debug visualization (PROVIDED)
            self._update_debug_display(frame, steering_angle, debug_level)
            
            return steering_angle
            
        except Exception as e:
            print(f"Processing error: {e}")
            return 0.0
    
    # =========================================================================
    # STUDENT IMPLEMENTATION SECTION - COMPLETE THESE 4 FUNCTIONS
    # =========================================================================
    
    def _extract_roi(self, frame):
        """
        STUDENT TODO: Extract Region of Interest from the image
        
        Your task:
        1. Calculate pixel coordinates from the percentage parameters above
        2. Use array slicing to crop the image: frame[top:bottom, left:right]
        3. Store the result in self.current_roi
        4. Store the bounds in self.roi_bounds for debug visualization
        
        Example:
            height, width = frame.shape[:2]
            top_px = int(height * self.roi_top)
            # ... calculate other bounds
            self.current_roi = frame[top_px:bottom_px, left_px:right_px]
        """
        
        # STUDENT CODE HERE:
        # Remove this dummy implementation and write your own
        
        height, width = frame.shape[:2]
        
        # TODO: Calculate pixel coordinates from percentages
        # top_px = int(height * self.roi_top)
        # bottom_px = int(height * self.roi_bottom)  
        # left_px = int(width * self.roi_left)
        # right_px = int(width * self.roi_right)
        
        # TODO: Extract ROI using array slicing
        # self.current_roi = frame[top_px:bottom_px, left_px:right_px]
        # self.roi_bounds = (top_px, bottom_px, left_px, right_px)
        
        # DUMMY IMPLEMENTATION (REPLACE THIS):
        self.current_roi = frame  # Just use full frame for now
        self.roi_bounds = (0, height, 0, width)
        
        # Update implementation status
        # Set this to True when you implement the function properly
        self.current_debug_data['implementation_status']['roi_extraction'] = False
    
    def _convert_to_grayscale(self):
        """
        STUDENT TODO: Convert the ROI to grayscale
        
        Your task:
        1. Check if self.current_roi exists and is not None
        2. Convert from BGR color to grayscale
        3. Store result in self.gray_image
        
        Methods you can use:
        - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        - Or manual conversion: gray = 0.299*R + 0.587*G + 0.114*B
        
        Example:
            if self.current_roi is not None:
                self.gray_image = cv2.cvtColor(self.current_roi, cv2.COLOR_BGR2GRAY)
        """
        
        # STUDENT CODE HERE:
        
        # TODO: Convert ROI to grayscale
        # if self.current_roi is not None:
        #     self.gray_image = cv2.cvtColor(self.current_roi, cv2.COLOR_BGR2GRAY)
        
        # DUMMY IMPLEMENTATION (REPLACE THIS):
        if self.current_roi is not None:
            # Just make a black image for now
            height, width = self.current_roi.shape[:2]
            self.gray_image = np.zeros((height, width), dtype=np.uint8)
        
        # Update implementation status
        # Set this to True when you implement the function properly
        self.current_debug_data['implementation_status']['grayscale_conversion'] = False
    
    def _detect_line_center(self):
        """
        STUDENT TODO: Find the center of the line in the grayscale image
        
        Your task:
        1. Use edge detection to find the line (cv2.Canny is recommended)
        2. Find the center point of the detected line
        3. Store coordinates in self.line_center_x and self.line_center_y
        
        Suggested approach:
        1. Apply Canny edge detection: cv2.Canny(gray, low_thresh, high_thresh)
        2. Find white pixels: np.where(edges > 0)
        3. Calculate mean position of white pixels
        
        Example:
            if self.gray_image is not None:
                edges = cv2.Canny(self.gray_image, self.canny_low, self.canny_high)
                white_pixels = np.where(edges > 0)
                if len(white_pixels[0]) > 0:
                    self.line_center_y = int(np.mean(white_pixels[0]))
                    self.line_center_x = int(np.mean(white_pixels[1]))
        """
        
        # STUDENT CODE HERE:
        
        # TODO: Apply edge detection and find line center
        # if self.gray_image is not None:
        #     edges = cv2.Canny(self.gray_image, self.canny_low, self.canny_high)
        #     white_pixels = np.where(edges > 0)
        #     if len(white_pixels[0]) > 0:
        #         self.line_center_y = int(np.mean(white_pixels[0]))
        #         self.line_center_x = int(np.mean(white_pixels[1]))
        
        # DUMMY IMPLEMENTATION (REPLACE THIS):
        if self.gray_image is not None:
            height, width = self.gray_image.shape
            # Just put line center in middle of image for now
            self.line_center_x = width // 2
            self.line_center_y = height // 2
        
        # Update implementation status
        # Set this to True when you implement the function properly
        self.current_debug_data['implementation_status']['line_detection'] = False
    
    def _calculate_error(self):
        """
        STUDENT TODO: Calculate steering error
        
        Your task:
        1. Find the center X coordinate of the ROI image
        2. Calculate error as: image_center_x - line_center_x
        3. Store result in self.current_error
        
        The error tells us:
        - Positive error: line is to the LEFT of center → steer RIGHT
        - Negative error: line is to the RIGHT of center → steer LEFT  
        - Zero error: line is centered → go straight
        
        Example:
            if self.current_roi is not None and self.line_center_x is not None:
                roi_width = self.current_roi.shape[1]
                self.image_center_x = roi_width // 2
                self.current_error = self.image_center_x - self.line_center_x
        """
        
        # STUDENT CODE HERE:
        
        # TODO: Calculate error between image center and line center
        # if self.current_roi is not None and self.line_center_x is not None:
        #     roi_width = self.current_roi.shape[1]  
        #     self.image_center_x = roi_width // 2
        #     self.current_error = self.image_center_x - self.line_center_x
        
        # DUMMY IMPLEMENTATION (REPLACE THIS):
        if self.current_roi is not None:
            roi_width = self.current_roi.shape[1]
            self.image_center_x = roi_width // 2
            # Dummy error - just use 0 for now
            self.current_error = 0.0
        
        # Update implementation status
        # Set this to True when you implement the function properly
        self.current_debug_data['implementation_status']['error_calculation'] = False
    
    # =========================================================================
    # PROVIDED FUNCTIONS - Students don't need to modify these
    # =========================================================================
    
    def _apply_pid_control(self):
        """Apply PID control to convert error to steering angle (PROVIDED)"""
        
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            dt = 0.1
        
        # PID calculation
        proportional = self.Kp * self.current_error
        
        self.integral_error += self.current_error * dt
        integral = self.Ki * self.integral_error
        
        derivative_error = (self.current_error - self.previous_error) / dt
        derivative = self.Kd * derivative_error
        
        # Combine terms and limit output
        steering_angle = proportional + integral + derivative
        steering_angle = np.clip(steering_angle, -45, 45)
        
        # Update state
        self.previous_error = self.current_error
        self.last_time = current_time
        
        return steering_angle
    
    def _update_debug_display(self, original_frame, steering_angle, debug_level):
        """Update debug visualization and data (PROVIDED)"""
        
        # Update sidebar debug data
        lines_detected = 1 if (self.line_center_x is not None and 
                              self.current_debug_data['implementation_status']['line_detection']) else 0
        
        self.current_debug_data.update({
            'error_px': round(self.current_error, 1),
            'steering_angle': round(steering_angle, 1),
            'lines_detected': lines_detected
        })
        
        # Create debug frame
        if debug_level > 0:
            self.debug_frame = self._create_debug_visualization(
                original_frame, steering_angle, debug_level
            )
        else:
            self.debug_frame = original_frame.copy()
    
    def _create_debug_visualization(self, frame, steering_angle, debug_level):
        """Create visual debug overlay (PROVIDED)"""
        
        debug_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Level 1: Show implementation status
        if debug_level >= 1:
            status_y = 30
            for func_name, implemented in self.current_debug_data['implementation_status'].items():
                color = (0, 255, 0) if implemented else (0, 0, 255)
                status = "✓" if implemented else "✗"
                text = f"{status} {func_name.replace('_', ' ').title()}"
                cv2.putText(debug_frame, text, (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                status_y += 20
            
            # Show current values
            values_text = [
                f"Error: {self.current_error:.1f}px",
                f"Steering: {steering_angle:.1f}°"
            ]
            
            for i, text in enumerate(values_text):
                cv2.putText(debug_frame, text, (10, height - 40 + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Level 2: Show ROI bounds
        if debug_level >= 2 and len(self.roi_bounds) == 4:
            top, bottom, left, right = self.roi_bounds
            cv2.rectangle(debug_frame, (left, top), (right, bottom), (0, 255, 255), 2)
            cv2.putText(debug_frame, "ROI", (left, top-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Level 3: Show detected centers
        if debug_level >= 3:
            # Show image center (blue)
            if self.image_center_x is not None and len(self.roi_bounds) == 4:
                top, bottom, left, right = self.roi_bounds
                center_global_x = left + self.image_center_x
                center_global_y = (top + bottom) // 2
                cv2.circle(debug_frame, (center_global_x, center_global_y), 8, (255, 0, 0), -1)
                cv2.putText(debug_frame, "IMG CENTER", 
                           (center_global_x-40, center_global_y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # Show line center (green)
            if (self.line_center_x is not None and self.line_center_y is not None and 
                len(self.roi_bounds) == 4):
                top, bottom, left, right = self.roi_bounds
                line_global_x = left + self.line_center_x
                line_global_y = top + self.line_center_y
                cv2.circle(debug_frame, (line_global_x, line_global_y), 8, (0, 255, 0), -1)
                cv2.putText(debug_frame, "LINE CENTER", 
                           (line_global_x-40, line_global_y+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Draw error line (red)
                if self.image_center_x is not None:
                    cv2.line(debug_frame, 
                           (center_global_x, center_global_y),
                           (line_global_x, line_global_y), 
                           (0, 0, 255), 2)
        
        return debug_frame
    
    def get_debug_frame(self):
        """Return current debug frame"""
        return self.debug_frame
    
    def update_parameters(self, **kwargs):
        """Update parameters during runtime"""
        if 'kp' in kwargs and kwargs['kp'] is not None:
            self.Kp = kwargs['kp']
        if 'ki' in kwargs and kwargs['ki'] is not None:
            self.Ki = kwargs['ki']
        if 'kd' in kwargs and kwargs['kd'] is not None:
            self.Kd = kwargs['kd']

# =========================================================================
# STUDENT TESTING SECTION
# =========================================================================

def test_implementation():
    """Test your implementation step by step"""
    
    print("🧪 Testing Line Follower Implementation")
    print("="*50)
    
    # Create test instance
    lf = LineFollower()
    
    # Create test image with white line
    test_image = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.line(test_image, (100, 50), (100, 200), (255, 255, 255), 10)
    
    print("📸 Processing test image...")
    
    # Test each function step by step
    print("\n1. Testing ROI extraction...")
    lf._extract_roi(test_image)
    roi_status = "✅ Working" if lf.current_roi is not None else "❌ Not implemented"
    print(f"   ROI: {roi_status}")
    
    print("\n2. Testing grayscale conversion...")
    lf._convert_to_grayscale()
    gray_status = "✅ Working" if lf.gray_image is not None else "❌ Not implemented"
    print(f"   Grayscale: {gray_status}")
    
    print("\n3. Testing line detection...")
    lf._detect_line_center()
    line_status = "✅ Working" if lf.line_center_x is not None else "❌ Not implemented"
    print(f"   Line detection: {line_status}")
    
    print("\n4. Testing error calculation...")
    lf._calculate_error()
    error_status = "✅ Working" if lf.image_center_x is not None else "❌ Not implemented"
    print(f"   Error calculation: {error_status}")
    
    # Test full pipeline
    print("\n5. Testing full pipeline...")
    steering = lf.compute_steering_angle(test_image, debug_level=1)
    print(f"   Steering angle: {steering:.1f}°")
    
    print(f"\n📊 Implementation status:")
    for func, status in lf.current_debug_data['implementation_status'].items():
        icon = "✅" if status else "⭕"
        print(f"   {icon} {func.replace('_', ' ').title()}")
    
    print(f"\n🎯 Next steps:")
    print("1. Implement the functions marked with ⭕")
    print("2. Set implementation_status to True when complete")  
    print("3. Test with real camera via web interface")
    print("4. Tune parameters for better performance")

if __name__ == "__main__":
    test_implementation()