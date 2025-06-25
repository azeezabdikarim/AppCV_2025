#!/usr/bin/env python3

"""
Line Following Module - Week 1 Assignment
=========================================

This module provides the foundation for line following with computer vision.
Students will complete the TODO sections to implement:
1. ROI (Region of Interest) tuning
2. Line detection with image processing
3. Error calculation (image center - line center)
4. PID parameter tuning

The debug system works immediately - you'll see visual feedback even before
completing the implementation.
"""

import cv2
import numpy as np
import time

class LineFollower:
    def __init__(self):
        """Initialize the line follower with default parameters"""
        
        # =================================================================
        # STUDENT TUNING SECTION - MODIFY THESE VALUES
        # =================================================================
        
        # ROI (Region of Interest) - Focus on road area
        # TODO: Adjust these values to crop the image to just the road area
        # Start with full frame, then gradually shrink to focus on the line
        self.roi_top_offset = 0.3      # 0.0 = top of image, 1.0 = bottom
        self.roi_bottom_offset = 0.3   # portion of image height to include
        self.roi_left_offset = 0.1     # portion of image width to start from
        self.roi_right_offset = 0.9    # portion of image width to end at
        
        # Image processing parameters
        # TODO: Experiment with these values for better line detection
        self.blur_kernel_size = 5      # Must be odd number (3, 5, 7, 9...)
        self.canny_low_threshold = 50  # Lower values = more edges detected
        self.canny_high_threshold = 150 # Higher values = only strong edges
        
        # PID Controller parameters  
        # TODO: Tune these for smooth steering response
        self.Kp = 0.8    # Proportional gain - how strongly to respond to current error
        self.Ki = 0.1    # Integral gain - how strongly to respond to accumulated error
        self.Kd = 0.3    # Derivative gain - how strongly to respond to error changes
        
        # Speed control
        self.base_speed = 100  # Base forward speed (0-255)
        
        # =================================================================
        # SYSTEM VARIABLES - Don't modify these directly
        # =================================================================
        
        # PID state variables
        self.previous_error = 0.0
        self.integral_error = 0.0
        self.last_time = time.time()
        
        # Debug and visualization
        self.debug_frame = None
        self.current_debug_data = {
            'error_px': 0.0,
            'steering_angle': 0.0,
            'lines_detected': 0,
            'roi_points': [],
            'line_center': None,
            'image_center': None
        }
        
        print("✅ Line follower initialized")
        print("🎯 TODO: Tune ROI parameters, image processing, and PID gains")
    
    def compute_steering_angle(self, frame, debug_level=0):
        """
        Main function: compute steering angle from camera frame
        
        Args:
            frame: Camera image (BGR format)
            debug_level: 0-4, higher = more debug visualization
            
        Returns:
            steering_angle: Float, degrees to steer (-90 to +90)
        """
        if frame is None:
            return 0.0
        
        # Step 1: Create ROI (Region of Interest)
        roi_frame, roi_points = self._create_roi(frame)
        
        # Step 2: Process image to detect lines
        line_center = self._detect_line_center(roi_frame)
        
        # Step 3: Calculate error (image center - line center)
        error_px = self._calculate_error(roi_frame, line_center)
        
        # Step 4: Apply PID control
        steering_angle = self._apply_pid_control(error_px)
        
        # Step 5: Update debug information
        self._update_debug_data(frame, roi_points, line_center, error_px, steering_angle, debug_level)
        
        return steering_angle
    
    def _create_roi(self, frame):
        """
        Create Region of Interest to focus on road area
        
        TODO SECTION FOR STUDENTS:
        The ROI parameters at the top of this file control which part of the 
        image we focus on. Experiment with:
        - roi_top_offset: How much of the top to crop out (sky, horizon)
        - roi_bottom_offset: How much of the bottom to include  
        - roi_left_offset, roi_right_offset: Horizontal cropping
        
        Good values depend on camera mounting angle and field of view.
        """
        height, width = frame.shape[:2]
        
        # Calculate ROI bounds based on student parameters
        top_y = int(height * self.roi_top_offset)
        bottom_y = int(height * self.roi_bottom_offset)
        left_x = int(width * self.roi_left_offset)
        right_x = int(width * self.roi_right_offset)
        
        # Create ROI points for visualization
        roi_points = [
            (left_x, top_y),      # Top-left
            (right_x, top_y),     # Top-right  
            (right_x, bottom_y),  # Bottom-right
            (left_x, bottom_y)    # Bottom-left
        ]
        
        # Extract ROI from frame
        roi_frame = frame[top_y:bottom_y, left_x:right_x]
        
        return roi_frame, roi_points
    
    def _detect_line_center(self, roi_frame):
        """
        Detect the center of the line in the ROI
        
        TODO SECTION FOR STUDENTS:
        This is where you implement line detection using computer vision.
        The basic pipeline is:
        1. Convert to grayscale
        2. Apply blur to reduce noise  
        3. Use Canny edge detection
        4. Find contours or use Hough lines
        5. Calculate the center point of detected lines
        
        Sample code snippets are provided below - modify and use them!
        """
        
        if roi_frame.size == 0:
            return None
            
        try:
            # TODO STUDENT IMPLEMENTATION:
            # Uncomment and modify the code below for line detection
            
            # SAMPLE CODE - BASIC APPROACH:
            # Convert to grayscale
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, self.canny_low_threshold, self.canny_high_threshold)
            
            # TODO: Implement line detection here
            # APPROACH 1 - Contour method:
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour (assuming it's the line)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate centroid
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
            
            # APPROACH 2 - Simple center-of-mass method:
            # Find all white pixels and calculate their center
            # white_pixels = np.where(edges > 0)
            # if len(white_pixels[0]) > 0:
            #     center_y = int(np.mean(white_pixels[0]))
            #     center_x = int(np.mean(white_pixels[1]))
            #     return (center_x, center_y)
            
            # APPROACH 3 - Hough Lines (more advanced):
            # lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
            #                        minLineLength=30, maxLineGap=5)
            # if lines is not None:
            #     # Process lines and find center
            #     pass
            
            return None  # No line detected
            
        except Exception as e:
            print(f"Line detection error: {e}")
            return None
    
    def _calculate_error(self, roi_frame, line_center):
        """
        Calculate error as difference between image center and line center
        
        This is the key insight for line following:
        - If line is to the left of center: negative error → steer right
        - If line is to the right of center: positive error → steer left
        - If line is centered: zero error → go straight
        
        TODO SECTION FOR STUDENTS:
        The basic calculation is provided, but you might want to experiment with:
        - Using only horizontal error (x-direction) vs. both x and y
        - Weighting the error differently
        - Using multiple points along the line
        """
        
        if roi_frame.size == 0:
            return 0.0
            
        # Calculate image center
        height, width = roi_frame.shape[:2]
        image_center_x = width / 2
        image_center_y = height / 2
        
        # Store for debug visualization
        self.current_debug_data['image_center'] = (int(image_center_x), int(image_center_y))
        
        if line_center is None:
            # No line detected - return moderate error to encourage searching
            return 0.0
        
        # Store line center for debug
        self.current_debug_data['line_center'] = line_center
        
        # TODO: Calculate error
        # Basic approach: horizontal difference
        line_center_x = line_center[0]
        error_px = image_center_x - line_center_x
        
        # Optional: Include vertical component
        # line_center_y = line_center[1] 
        # vertical_error = image_center_y - line_center_y
        # error_px = math.sqrt(error_px**2 + vertical_error**2)
        
        return error_px
    
    def _apply_pid_control(self, error_px):
        """
        Apply PID control to convert error to steering angle
        
        PID Control combines:
        - P (Proportional): React to current error
        - I (Integral): React to accumulated past errors  
        - D (Derivative): React to rate of error change
        
        TODO SECTION FOR STUDENTS:
        The PID implementation is provided, but you need to tune the gains:
        - Kp: Start with 0.5-1.0, increase for stronger response
        - Ki: Start with 0.0-0.2, helps eliminate steady-state error
        - Kd: Start with 0.0-0.5, helps reduce oscillations
        """
        
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            dt = 0.1  # Prevent division by zero
        
        # Proportional term
        proportional = self.Kp * error_px
        
        # Integral term (accumulated error over time)
        self.integral_error += error_px * dt
        integral = self.Ki * self.integral_error
        
        # Derivative term (rate of error change)
        derivative_error = (error_px - self.previous_error) / dt
        derivative = self.Kd * derivative_error
        
        # Combine PID terms
        steering_angle = proportional + integral + derivative
        
        # Limit output to reasonable steering range
        steering_angle = np.clip(steering_angle, -45, 45)
        
        # Update state for next iteration
        self.previous_error = error_px
        self.last_time = current_time
        
        return steering_angle
    
    def _update_debug_data(self, original_frame, roi_points, line_center, error_px, steering_angle, debug_level):
        """
        Update debug visualization and data
        This function creates the visual overlay you see in the web interface
        """
        
        # Update debug data for sidebar
        self.current_debug_data.update({
            'error_px': round(error_px, 1),
            'steering_angle': round(steering_angle, 1),
            'lines_detected': 1 if line_center else 0,
            'roi_points': roi_points
        })
        
        # Create debug frame for visualization
        if debug_level > 0:
            self.debug_frame = self._create_debug_visualization(
                original_frame, roi_points, line_center, error_px, steering_angle, debug_level
            )
        else:
            self.debug_frame = original_frame.copy()
    
    def _create_debug_visualization(self, frame, roi_points, line_center, error_px, steering_angle, debug_level):
        """Create debug visualization overlay"""
        
        debug_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Level 1: Basic ROI and status
        if debug_level >= 1:
            # Draw ROI rectangle
            if len(roi_points) == 4:
                roi_array = np.array(roi_points, np.int32)
                cv2.polylines(debug_frame, [roi_array], True, (0, 255, 255), 2)
                cv2.putText(debug_frame, "ROI", roi_points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Status text
            status = f"Error: {error_px:.1f}px | Steer: {steering_angle:.1f}°"
            cv2.putText(debug_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Level 2: Line center and image center
        if debug_level >= 2 and len(roi_points) == 4:
            roi_offset = roi_points[0]  # Top-left corner offset
            
            # Draw image center (in ROI coordinates)
            if self.current_debug_data['image_center']:
                img_center = self.current_debug_data['image_center']
                center_global = (img_center[0] + roi_offset[0], img_center[1] + roi_offset[1])
                cv2.circle(debug_frame, center_global, 8, (255, 0, 0), -1)
                cv2.putText(debug_frame, "IMG CENTER", (center_global[0]-50, center_global[1]-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw detected line center
            if line_center:
                line_global = (line_center[0] + roi_offset[0], line_center[1] + roi_offset[1])
                cv2.circle(debug_frame, line_global, 8, (0, 255, 0), -1)
                cv2.putText(debug_frame, "LINE CENTER", (line_global[0]-50, line_global[1]+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw error line
                if self.current_debug_data['image_center']:
                    cv2.line(debug_frame, center_global, line_global, (255, 255, 0), 2)
        
        # Level 3: PID component breakdown
        if debug_level >= 3:
            p_term = self.Kp * error_px
            i_term = self.Ki * self.integral_error
            d_term = self.Kd * (error_px - self.previous_error) / 0.1
            
            pid_text = [
                f"P: {p_term:.1f}° (Kp={self.Kp})",
                f"I: {i_term:.1f}° (Ki={self.Ki})", 
                f"D: {d_term:.1f}° (Kd={self.Kd})",
                f"Total: {steering_angle:.1f}°"
            ]
            
            for i, text in enumerate(pid_text):
                cv2.putText(debug_frame, text, (10, 60 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Level 4: Parameter display
        if debug_level >= 4:
            param_text = [
                f"ROI: T{self.roi_top_offset:.1f} B{self.roi_bottom_offset:.1f}",
                f"Blur: {self.blur_kernel_size}",
                f"Canny: {self.canny_low_threshold}-{self.canny_high_threshold}"
            ]
            
            for i, text in enumerate(param_text):
                cv2.putText(debug_frame, text, (width-200, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return debug_frame
    
    def get_debug_frame(self):
        """Return the current debug frame for display"""
        return self.debug_frame
    
    def update_parameters(self, **kwargs):
        """Update parameters during runtime (called from web interface)"""
        if 'kp' in kwargs and kwargs['kp'] is not None:
            self.Kp = kwargs['kp']
        if 'ki' in kwargs and kwargs['ki'] is not None:
            self.Ki = kwargs['ki']  
        if 'kd' in kwargs and kwargs['kd'] is not None:
            self.Kd = kwargs['kd']
        
        print(f"🔧 PID parameters updated: Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}")

# =============================================================================
# STUDENT TESTING SECTION
# =============================================================================

def test_line_follower():
    """
    Simple test function for students to validate their implementation
    Run this with: python3 line_follower.py
    """
    
    print("🧪 Testing Line Follower Implementation")
    print("="*50)
    
    # Create test instance
    lf = LineFollower()
    
    # Create a simple test image with a white line
    test_image = np.zeros((240, 320, 3), dtype=np.uint8)
    
    # Draw a white line (slightly off-center to create error)
    cv2.line(test_image, (140, 50), (140, 200), (255, 255, 255), 10)
    
    print("📸 Processing test image...")
    steering_angle = lf.compute_steering_angle(test_image, debug_level=2)
    
    print(f"✅ Steering angle computed: {steering_angle:.1f}°")
    print(f"📊 Debug data: {lf.current_debug_data}")
    
    # Test different scenarios
    scenarios = [
        ("Line left of center", (100, 120)),
        ("Line right of center", (220, 120)), 
        ("Line centered", (160, 120))
    ]
    
    for desc, line_pos in scenarios:
        test_img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.line(test_img, (line_pos[0], 50), (line_pos[0], 200), (255, 255, 255), 10)
        
        angle = lf.compute_steering_angle(test_img, debug_level=0)
        print(f"📋 {desc}: {angle:.1f}°")
    
    print("\n🎯 TODO for students:")
    print("1. Tune ROI parameters in __init__()")
    print("2. Implement line detection in _detect_line_center()")
    print("3. Tune PID parameters for smooth control")
    print("4. Test with real camera feed via web interface")

if __name__ == "__main__":
    test_line_follower()