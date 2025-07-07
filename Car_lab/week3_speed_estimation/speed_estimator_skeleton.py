#!/usr/bin/env python3

import cv2
import numpy as np
import json
import os

class SpeedEstimator:
    """
    Week 3 Implementation: Optical Flow Speed Estimation
    
    Students will implement optical flow analysis to estimate robot speed
    from camera frames, following the Lucas-Kanade method from the academic paper.
    """
    
    def __init__(self):
        """Initialize optical flow parameters and load calibration"""
        
        # Optical flow parameters (students may need to tune these)
        self.feature_params = {
            'maxCorners': 100,      # Maximum number of features to track
            'qualityLevel': 0.3,    # Quality level for corner detection
            'minDistance': 7,       # Minimum distance between features
            'blockSize': 7          # Size of averaging block for corner detection
        }
        
        self.lk_params = {
            'winSize': (15, 15),    # Window size for Lucas-Kanade
            'maxLevel': 2,          # Maximum pyramid levels
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }
        
        # Speed calibration parameters (loaded from calibration file)
        self.calibration_loaded = False
        self.flow_to_speed_slope = 1.0
        self.flow_to_speed_intercept = 0.0
        
        # Internal state for tracking
        self.previous_gray = None
        self.previous_features = None

        self.speed_history = []
        self.max_history_length = 10

        # Load calibration parameters
        self._load_calibration()
        
        print("SpeedEstimator initialized - Students: Implement estimate_speed()!")
    
    def _load_calibration(self):
        """Load speed calibration parameters from file"""
        try:
            calibration_file = os.path.join(os.path.dirname(__file__), 'calibration_params.json')
            
            if os.path.exists(calibration_file):
                with open(calibration_file, 'r') as f:
                    params = json.load(f)
                
                self.flow_to_speed_slope = params.get('slope', 1.0)
                self.flow_to_speed_intercept = params.get('intercept', 0.0)
                self.calibration_loaded = True
                
                print(f"✅ Calibration loaded: slope={self.flow_to_speed_slope:.3f}, intercept={self.flow_to_speed_intercept:.3f}")
            else:
                print(f"⚠️  No calibration file found: {calibration_file}")
                print("   Using default parameters. Run speed calibration first!")
                
        except Exception as e:
            print(f"❌ Error loading calibration: {e}")
    
    def estimate_speed(self, current_frame, previous_frame):
        """
        Main speed estimation function students must implement
        
        Args:
            current_frame: numpy array of shape (240, 320, 3) - current RGB image
            previous_frame: numpy array of shape (240, 320, 3) - previous RGB image
            
        Returns:
            speed: float - estimated speed in units/second (calibrated)
        """
        
        if previous_frame is None:
            return 0.0
        
        try:
            # =============================================================
            # STEP 1: CONVERT TO GRAYSCALE (following paper section 1.3)
            # Optical flow works on grayscale images
            # =============================================================
            
            # STUDENTS IMPLEMENT:
            # Convert both frames to grayscale using cv2.cvtColor()
            
            current_gray = None   # TODO: Convert current frame to grayscale
            previous_gray = None  # TODO: Convert previous frame to grayscale
            
            # =============================================================
            # STEP 2: FEATURE DETECTION (following paper)
            # Find good features to track in the previous frame
            # =============================================================
            
            # STUDENTS IMPLEMENT:
            # Use cv2.goodFeaturesToTrack() to find corner features
            # This identifies distinct points that can be reliably tracked
            
            features_prev = None  # TODO: Detect features in previous frame
            
            if features_prev is None or len(features_prev) == 0:
                return 0.0  # No features to track
            
            # =============================================================
            # STEP 3: OPTICAL FLOW CALCULATION (following paper section 1.1)
            # Use Lucas-Kanade method to track features between frames
            # =============================================================
            
            # STUDENTS IMPLEMENT:
            # Use cv2.calcOpticalFlowPyrLK() to track features from previous to current frame
            # This implements the Lucas-Kanade algorithm with pyramids
            
            features_curr = None  # TODO: Track features using Lucas-Kanade
            status = None         # TODO: Get tracking status for each feature
            error = None          # TODO: Get tracking error for each feature
            
            # =============================================================
            # STEP 4: FILTER GOOD FEATURES
            # Only use features that were successfully tracked
            # =============================================================
            
            if features_curr is None or status is None:
                return 0.0
            
            # STUDENTS IMPLEMENT:
            # Filter out features where tracking failed (status == 0)
            # Keep only good features for flow calculation
            
            good_prev = []  # TODO: Filter previous features
            good_curr = []  # TODO: Filter current features
            
            # for i, (status_flag, err) in enumerate(zip(status, error)):
            #     if status_flag == 1:  # Good tracking
            #         good_prev.append(features_prev[i])
            #         good_curr.append(features_curr[i])
            
            if len(good_prev) < 10:  # Need minimum features for reliable estimation
                return 0.0
            
            # =============================================================
            # STEP 5: CALCULATE FLOW MAGNITUDE
            # Compute the magnitude of optical flow vectors
            # =============================================================
            
            # STUDENTS IMPLEMENT:
            # For each tracked feature, calculate the displacement vector
            # Compute the magnitude (length) of each displacement
            # Take the average magnitude as the overall flow measure
            
            flow_magnitudes = []
            
            # TODO: Calculate flow magnitude for each good feature pair
            # for prev_pt, curr_pt in zip(good_prev, good_curr):
            #     dx = curr_pt[0] - prev_pt[0]
            #     dy = curr_pt[1] - prev_pt[1]
            #     magnitude = np.sqrt(dx*dx + dy*dy)
            #     flow_magnitudes.append(magnitude)
            
            if not flow_magnitudes:
                return 0.0
            
            # Average flow magnitude
            avg_flow_magnitude = np.mean(flow_magnitudes)
            
            # =============================================================
            # STEP 6: CONVERT FLOW TO SPEED (using calibration)
            # Apply calibration parameters to get real-world speed
            # =============================================================
            
            # STUDENTS IMPLEMENT:
            # Apply linear calibration: speed = slope * flow + intercept
            # This converts pixel flow to real-world speed units
            
            if self.calibration_loaded:
                speed = self.flow_to_speed_slope * avg_flow_magnitude + self.flow_to_speed_intercept
            else:
                # Without calibration, just return raw flow magnitude
                speed = avg_flow_magnitude
            
            # Ensure speed is non-negative
            speed = max(0.0, speed)

            self.speed_history.append(speed)
            if len(self.speed_history) > self.max_history_length:
                self.speed_history.pop(0)
            
            return float(speed)
            
        except Exception as e:
            print(f"Speed estimation error: {e}")
            return 0.0
    
    def is_calibrated(self):
        """Check if speed calibration parameters are available"""
        return self.calibration_loaded
    
    def get_speed_history(self):
        """Get speed history for web interface and smoothing"""
        if not self.speed_history:
            return {
                'current_speed': 0.0,
                'smoothed_speed': 0.0,
                'speed_history': [],
                'avg_speed': 0.0
            }
        
        # Calculate smoothed speed (average of last 3 readings)
        smoothing_window = min(3, len(self.speed_history))
        recent_speeds = self.speed_history[-smoothing_window:]
        smoothed_speed = sum(recent_speeds) / len(recent_speeds)
        
        return {
            'current_speed': self.speed_history[-1],
            'smoothed_speed': smoothed_speed,
            'speed_history': self.speed_history.copy(),
            'avg_speed': sum(self.speed_history) / len(self.speed_history)
        }

# =============================================================================
# HELPFUL REFERENCE CODE (for students to adapt)
# =============================================================================

"""
EXAMPLE FEATURE DETECTION:
features = cv2.goodFeaturesToTrack(
    gray_image,
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

EXAMPLE OPTICAL FLOW:
features_next, status, error = cv2.calcOpticalFlowPyrLK(
    old_gray, 
    new_gray, 
    features_prev, 
    None, 
    **lk_params
)

EXAMPLE FLOW MAGNITUDE CALCULATION:
for i, (prev_pt, curr_pt) in enumerate(zip(good_prev, good_curr)):
    if status[i] == 1:  # Successfully tracked
        dx = curr_pt[0][0] - prev_pt[0][0]
        dy = curr_pt[0][1] - prev_pt[0][1]
        magnitude = np.sqrt(dx*dx + dy*dy)
        flow_magnitudes.append(magnitude)

avg_flow = np.mean(flow_magnitudes)
"""