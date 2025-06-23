#!/usr/bin/env python3

import cv2
import numpy as np
import time

class LineFollower:
    """
    COMPLETE SAMPLE SOLUTION: Week 1 Line Following Implementation
    
    This solution demonstrates all concepts from the lab:
    - Computer vision pipeline (ROI, Canny, Hough transform)
    - PID controller with all three terms
    - Comprehensive debug visualization
    - Robust error handling
    """
    
    def __init__(self):
        """Initialize PID controller and line detection parameters"""
        
        # =================================================================
        # PID Controller Parameters (Tuned for smooth performance)
        # =================================================================
        self.Kp = 0.8  # Proportional gain - responsive but not oscillatory
        self.Ki = 0.02  # Integral gain - eliminates steady-state error
        self.Kd = 0.15  # Derivative gain - dampens oscillations
        
        # PID controller state
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()
        
        # Integral windup protection
        self.integral_limit = 100.0  # Prevent integral from growing too large
        
        # =================================================================
        # Computer Vision Parameters (Tuned for typical lighting)
        # =================================================================
        self.canny_low = 70    # Lower threshold for Canny edge detection
        self.canny_high = 270  # Upper threshold for Canny edge detection
        self.crop_offset_y = 180  # Start cropping 180px from top
        self.crop_height = 60     # Height of ROI (bottom 60 pixels)
        
        # Image center for error calculation (320px width / 2)
        self.image_center_x = 160
        
        # Steering limits (degrees) - PiCar-X servo limits
        self.max_steering_angle = 30
        
        # Hough transform parameters
        self.hough_threshold = 40      # Minimum intersections to form line
        self.hough_min_line_length = 40  # Minimum line length
        self.hough_max_line_gap = 15   # Maximum gap in line
        
        # Debug visualization state
        self.debug_frame = None
        
        # Performance monitoring
        self.processing_times = []
        self.lines_detected_history = []
        
        print("✅ LineFollower initialized with sample solution parameters")
    
    def compute_steering_angle(self, camera_frame, debug_level=0):
        """
        COMPLETE IMPLEMENTATION: Main line following algorithm
        
        This demonstrates the full pipeline from the lab documentation:
        1. Region of Interest extraction
        2. Canny edge detection
        3. Hough transform line detection
        4. Line center calculation
        5. PID control for steering
        """
        
        start_time = time.time()
        
        try:
            # Initialize debug frame if debugging enabled
            if debug_level > 0:
                self.debug_frame = camera_frame.copy()
            
            # =============================================================
            # STEP 1: REGION OF INTEREST (Section 1.1)
            # Extract bottom portion of image for line detection
            # =============================================================
            
            roi = camera_frame[self.crop_offset_y:self.crop_offset_y + self.crop_height, 0:320]
            
            # Debug Level 1+: Show ROI boundary
            if debug_level >= 1:
                cv2.rectangle(self.debug_frame, 
                            (0, self.crop_offset_y), 
                            (320, self.crop_offset_y + self.crop_height), 
                            (0, 255, 0), 2)
                cv2.putText(self.debug_frame, "ROI", (5, self.crop_offset_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # =============================================================
            # STEP 2: EDGE DETECTION (Section 1.2)
            # Apply Canny edge detection to find line boundaries
            # =============================================================
            
            # Convert to grayscale (required for Canny)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Optional: Apply Gaussian blur to reduce noise
            gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray_blurred, self.canny_low, self.canny_high)
            
            # Debug Level 2+: Show edge detection result in small inset
            if debug_level >= 2:
                edge_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                edge_resized = cv2.resize(edge_display, (160, 30))
                self.debug_frame[10:40, 10:170] = edge_resized
                cv2.rectangle(self.debug_frame, (10, 10), (170, 40), (255, 255, 0), 1)
                cv2.putText(self.debug_frame, "Canny Edges", (10, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # =============================================================
            # STEP 3: LINE DETECTION (Section 1.3)
            # Use Hough Transform to detect line segments
            # =============================================================
            
            lines = cv2.HoughLinesP(
                edges,
                rho=1,                              # Distance resolution: 1 pixel
                theta=np.pi/180,                    # Angle resolution: 1 degree
                threshold=self.hough_threshold,     # Minimum intersections
                minLineLength=self.hough_min_line_length,  # Minimum line length
                maxLineGap=self.hough_max_line_gap  # Maximum gap in line
            )
            
            # =============================================================
            # STEP 4: LINE CENTER CALCULATION (Section 1.4)
            # Calculate the center point of detected line segments
            # =============================================================
            
            line_center_x = self.image_center_x  # Default to center if no lines
            lines_found = 0
            
            if lines is not None and len(lines) > 0:
                # Filter lines by length and angle (optional improvement)
                valid_lines = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate line length
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    # Calculate line angle (optional: filter near-horizontal lines)
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    
                    # Accept lines that are reasonably long and not too steep
                    if length > 20 and abs(angle) < 60:  # Reasonable line criteria
                        valid_lines.append(line)
                
                if valid_lines:
                    # Calculate center points of all valid line segments
                    x_coords = []
                    
                    for line in valid_lines:
                        x1, y1, x2, y2 = line[0]
                        center_x = (x1 + x2) / 2
                        x_coords.append(center_x)
                        
                        # Debug Level 3+: Draw detected line segments
                        if debug_level >= 3:
                            # Draw line segment (adjust coordinates for ROI offset)
                            cv2.line(self.debug_frame, 
                                   (x1, y1 + self.crop_offset_y), 
                                   (x2, y2 + self.crop_offset_y), 
                                   (255, 0, 0), 2)
                            # Draw center point
                            center_y = (y1 + y2) / 2 + self.crop_offset_y
                            cv2.circle(self.debug_frame, 
                                     (int(center_x), int(center_y)), 
                                     3, (0, 0, 255), -1)
                    
                    # Calculate weighted average (or simple average)
                    line_center_x = np.mean(x_coords)
                    lines_found = len(valid_lines)
                else:
                    lines_found = 0
            
            # Store for performance monitoring
            self.lines_detected_history.append(lines_found)
            if len(self.lines_detected_history) > 50:
                self.lines_detected_history.pop(0)
            
            # =============================================================
            # STEP 5: PID CONTROL (Section 2)
            # Convert line position error to steering angle
            # =============================================================
            
            # Calculate error (horizontal distance from image center)
            error = line_center_x - self.image_center_x
            
            # Calculate time delta for integral and derivative terms
            current_time = time.time()
            dt = current_time - self.last_time
            
            if dt > 0:  # Avoid division by zero
                # Proportional term: immediate response to current error
                P = self.Kp * error
                
                # Integral term: accumulate error over time (with windup protection)
                self.integral += error * dt
                # Prevent integral windup
                self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
                I = self.Ki * self.integral
                
                # Derivative term: rate of change of error
                D = self.Kd * (error - self.last_error) / dt
                
                # Combined PID output
                pid_output = P + I + D
                
                # Convert to steering angle with limits
                steering_angle = np.clip(pid_output, -self.max_steering_angle, self.max_steering_angle)
                
                # Update for next iteration
                self.last_error = error
                self.last_time = current_time
                
                # =============================================================
                # DEBUG VISUALIZATION (All Levels)
                # =============================================================
                
                if debug_level >= 1:
                    # Draw image center line (yellow)
                    cv2.line(self.debug_frame, 
                           (self.image_center_x, self.crop_offset_y), 
                           (self.image_center_x, self.crop_offset_y + self.crop_height), 
                           (0, 255, 255), 2)
                    cv2.putText(self.debug_frame, "Center", 
                               (self.image_center_x + 5, self.crop_offset_y + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
                    # Draw detected line center (magenta)
                    center_y = self.crop_offset_y + self.crop_height//2
                    cv2.circle(self.debug_frame, 
                             (int(line_center_x), center_y), 
                             8, (255, 0, 255), -1)
                    cv2.putText(self.debug_frame, "Line Center", 
                               (int(line_center_x) + 10, center_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                    
                    # Draw error line (red)
                    cv2.line(self.debug_frame, 
                           (self.image_center_x, center_y), 
                           (int(line_center_x), center_y), 
                           (0, 0, 255), 3)
                    
                    # Add basic text overlays
                    cv2.putText(self.debug_frame, f"Error: {error:.1f}px", 
                               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(self.debug_frame, f"Steering: {steering_angle:.1f}deg", 
                               (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(self.debug_frame, f"Lines: {lines_found}", 
                               (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if debug_level >= 4:
                    # Show detailed PID component values
                    cv2.putText(self.debug_frame, f"P: {P:.2f} (Kp={self.Kp})", 
                               (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(self.debug_frame, f"I: {I:.2f} (Ki={self.Ki})", 
                               (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(self.debug_frame, f"D: {D:.2f} (Kd={self.Kd})", 
                               (200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(self.debug_frame, f"PID: {pid_output:.2f}", 
                               (200, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Show performance metrics
                    processing_time = (time.time() - start_time) * 1000
                    self.processing_times.append(processing_time)
                    if len(self.processing_times) > 20:
                        self.processing_times.pop(0)
                    
                    avg_processing_time = np.mean(self.processing_times)
                    cv2.putText(self.debug_frame, f"Proc: {avg_processing_time:.1f}ms", 
                               (200, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                return float(steering_angle)
            
            # Return 0 if timing calculation fails
            return 0.0
            
        except Exception as e:
            print(f"Line following error: {e}")
            # Safe fallback behavior
            if debug_level >= 1 and self.debug_frame is not None:
                cv2.putText(self.debug_frame, f"ERROR: {str(e)[:30]}", 
                           (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return 0.0  # Go straight when error occurs
    
    def get_debug_frame(self):
        """Return the debug visualization frame"""
        return self.debug_frame if self.debug_frame is not None else None
    
    def reset_integral(self):
        """Reset integral term (useful when starting or after errors)"""
        self.integral = 0.0
        print("🔄 PID integral term reset")
    
    def update_parameters(self, kp=None, ki=None, kd=None, canny_low=None, canny_high=None):
        """Update PID and vision parameters during runtime"""
        if kp is not None:
            self.Kp = kp
            print(f"✅ Kp updated to {kp}")
        if ki is not None:
            self.Ki = ki
            print(f"✅ Ki updated to {ki}")
        if kd is not None:
            self.Kd = kd
            print(f"✅ Kd updated to {kd}")
        if canny_low is not None:
            self.canny_low = canny_low
            print(f"✅ Canny low threshold updated to {canny_low}")
        if canny_high is not None:
            self.canny_high = canny_high
            print(f"✅ Canny high threshold updated to {canny_high}")
    
    def get_performance_stats(self):
        """Return performance statistics for monitoring"""
        return {
            'avg_processing_time_ms': np.mean(self.processing_times) if self.processing_times else 0,
            'avg_lines_detected': np.mean(self.lines_detected_history) if self.lines_detected_history else 0,
            'current_integral': self.integral,
            'last_error': self.last_error
        }
    
    def calibrate_for_lighting(self, test_frame):
        """
        BONUS FEATURE: Auto-calibrate Canny thresholds for current lighting
        Call this method with a sample frame to optimize edge detection
        """
        try:
            # Extract ROI for testing
            roi = test_frame[self.crop_offset_y:self.crop_offset_y + self.crop_height, 0:320]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Calculate image statistics
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Adaptive threshold calculation
            self.canny_low = max(30, mean_intensity - std_intensity)
            self.canny_high = min(300, mean_intensity + 2 * std_intensity)
            
            print(f"🔧 Auto-calibrated Canny thresholds: low={self.canny_low:.0f}, high={self.canny_high:.0f}")
            print(f"   Based on mean={mean_intensity:.0f}, std={std_intensity:.0f}")
            
        except Exception as e:
            print(f"❌ Auto-calibration failed: {e}")