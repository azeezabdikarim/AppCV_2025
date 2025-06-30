#!/usr/bin/env python3

import threading
import time
import cv2
import numpy as np
import sys
import os

# Add the parent directory to Python path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.movement_controls import MovementController
from utils.debug_visualizer import DebugVisualizer
from utils.utils import TimingUtils, CacheManager, StatusManager

try:
    from picarx import Picarx
except ImportError:
    print("WARNING: PicarX not available - running in simulation mode")
    Picarx = None

# =============================================================================
# EXPLICIT FEATURE CONTROL
# Students enable features when they're ready
# =============================================================================
FEATURES_ENABLED = {
    'line_following': False,   # Week 1 - Enable when ready
    'sign_detection': True,  # Week 2 - Student enables when implemented  
    'speed_estimation': False # Week 3 - Student enables when implemented
}

class RobotController:
    def __init__(self):
        """Initialize the robot controller"""
        try:
            # =================================================================
            # STUDENT TUNABLE PARAMETERS - Modify these as needed
            # =================================================================
            
            # Week 2: Object Detection & Depth Analysis Performance
            self.detection_interval = 0.5    # Run object detection every 0.5 seconds
            self.depth_interval = 1.0        # Run depth analysis every 1.0 seconds
            self.sign_stop_duration = 3.0    # Seconds to stop for detected signs
            self.stop_cooldown_duration = 2.0 # Cooldown after movement resumes
            
            # Debug and visualization settings
            self.debug_level = 0             # 0-4, higher = more debug info
            self.target_fps = 10             # Target frame rate for line following
            
            # =================================================================
            # SYSTEM VARIABLES - Don't modify these directly
            # =================================================================
            
            # Initialize hardware controller
            self.movement_controller = MovementController()
            
            # Initialize utility managers
            self.timing_utils = TimingUtils()
            self.cache_manager = CacheManager()
            self.status_manager = StatusManager()
            
            # Initialize debug visualizer
            self.debug_visualizer = DebugVisualizer()
            
            # Debug mode system
            self.debug_mode = "line_following"  # Default to Week 1
            self.available_modes = ["line_following", "object_detection", "speed_estimation", "full_system"]
            
            # Autonomous mode variables
            self.autonomous_mode = False
            self.frame_counter = 0
            self.previous_frame = None
            self.current_speed = 0.0
            self.sign_stop_until = None      # Time when sign stop expires
            
            # Performance tracking and timing
            self.last_frame_time = time.time()
            self.frame_interval = 1.0 / self.target_fps
            
            # Feature modules (loaded based on FEATURES_ENABLED)
            self.line_follower = None
            self.sign_detector = None
            self.speed_estimator = None
            
            # Feature status tracking
            self.feature_status = {
                'line_following': 'Disabled',
                'sign_detection': 'Disabled', 
                'speed_estimation': 'Disabled'
            }
            
            # Debug data for sidebar (clean, minimal)
            self.debug_data = {
                'error_px': 0.0,
                'steering_angle': 0.0,
                'lines_detected': 0,
                'mode': 'Manual'
            }
            
            # Load enabled features
            self._load_enabled_features()
            
            print("✅ Robot controller initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing robot: {e}")
    
    def _load_enabled_features(self):
        """Load only the features that are explicitly enabled"""
        
        print("Loading enabled features...")
        
        # Week 1: Line Following
        if FEATURES_ENABLED['line_following']:
            try:
                # Clear any cached modules to force reload
                module_name = 'week1_line_following.line_follower'
                if module_name in sys.modules:
                    del sys.modules[module_name]
                
                from week1_line_following.line_follower_easy_debugging import LineFollower
                self.line_follower = LineFollower()
                self.feature_status['line_following'] = 'Active'
                print("✅ Line following enabled and loaded")
                
            except Exception as e:
                self.feature_status['line_following'] = f'Error: {str(e)}'
                print(f"❌ Line following error: {e}")
        else:
            self.feature_status['line_following'] = 'Disabled'
            print("Line following disabled")
            
        # Week 2: Sign Detection
        if FEATURES_ENABLED['sign_detection']:
            try:
                if 'week2_object_detection.sign_detector' in sys.modules:
                    del sys.modules['week2_object_detection.sign_detector']
                    
                from week2_object_detection.sign_detector import SignDetector
                self.sign_detector = SignDetector()
                self.feature_status['sign_detection'] = 'Active'
                print("✅ Sign detection enabled and loaded")
            except Exception as e:
                self.feature_status['sign_detection'] = f'Error: {str(e)}'
                print(f"❌ Sign detection error: {e}")
        else:
            self.feature_status['sign_detection'] = 'Disabled'
            print("Sign detection disabled")
            
        # Week 3: Speed Estimation
        if FEATURES_ENABLED['speed_estimation']:
            try:
                if 'week3_speed_estimation.speed_estimator' in sys.modules:
                    del sys.modules['week3_speed_estimation.speed_estimator']
                    
                from week3_speed_estimation.speed_estimator import SpeedEstimator
                self.speed_estimator = SpeedEstimator()
                self.feature_status['speed_estimation'] = 'Active' 
                print("✅ Speed estimation enabled and loaded")
            except Exception as e:
                self.feature_status['speed_estimation'] = f'Error: {str(e)}'
                print(f"❌ Speed estimation error: {e}")
        else:
            self.feature_status['speed_estimation'] = 'Disabled'
            print("Speed estimation disabled")
        
        # Print final status
        print("Feature Status Summary:")
        for feature, status in self.feature_status.items():
            print(f"   {feature}: {status}")
    
    def start_autonomous_mode(self):
        """Start autonomous line following mode"""
        print("Starting autonomous mode...")
        
        if not FEATURES_ENABLED['line_following']:
            error_msg = "Line following is disabled in feature config"
            print(f"❌ {error_msg}")
            return False, error_msg
        
        if not self.line_follower:
            error_msg = f"Line following not available: {self.feature_status['line_following']}"
            print(f"❌ {error_msg}")
            return False, error_msg
            
        if not self.movement_controller.is_hardware_connected():
            error_msg = "Robot hardware not connected"
            print(f"❌ {error_msg}")
            return False, error_msg
            
        self.autonomous_mode = True
        self.frame_counter = 0
        self.debug_data['mode'] = 'Autonomous'
        print("✅ Autonomous mode started successfully")
        return True, "Autonomous mode started"
    
    def stop_autonomous_mode(self):
        """Stop autonomous mode and return to manual control"""
        self.autonomous_mode = False
        self.debug_data['mode'] = 'Manual'
        self.movement_controller.emergency_stop()
        print("Autonomous mode stopped")
        return True, "Autonomous mode stopped"
    
    def process_autonomous_frame(self, frame):
        """Main processing pipeline with clean debug separation"""
        
        display_frame = frame.copy()
        
        # Initialize debug data
        self.debug_data = {
            'error_px': 0.0,
            'steering_angle': 0.0,
            'lines_detected': 0,
            'mode': 'Autonomous' if self.autonomous_mode else 'Manual'
        }
        
        # Check if currently stopped for sign detection and cooldown logic
        current_time = time.time()
        stopped_for_sign = (self.sign_stop_until is not None and current_time < self.sign_stop_until)
        in_cooldown = self.status_manager.is_in_cooldown(current_time)
        
        # Week 2: Sign Detection (with timing and caching)
        if self.sign_detector and FEATURES_ENABLED['sign_detection'] and not stopped_for_sign and not in_cooldown:
            detected_signs = self._run_detection_with_timing(frame)
            if self.sign_detector.should_stop(detected_signs, frame):
                self.sign_stop_until = current_time + self.sign_stop_duration
                self.status_manager.set_recently_stopped(True)
                stopped_for_sign = True
                print("Stopping for detected sign")
        
        # Anti-infinite-stop: Reset cooldown when robot starts moving again
        if self.status_manager.recently_stopped_for_sign and not stopped_for_sign and self.autonomous_mode:
            self.status_manager.start_cooldown(current_time, self.stop_cooldown_duration)
            print("Stop cooldown activated")
        
        # Week 3: Speed Estimation
        if self.speed_estimator and FEATURES_ENABLED['speed_estimation']:
            self.current_speed = self.speed_estimator.estimate_speed(frame, self.previous_frame)
            self.debug_data['current_speed'] = round(self.current_speed, 1)
        
        # Week 1: Line Following (skip if stopped for sign)
        if self.line_follower and FEATURES_ENABLED['line_following']:
            try:
                steering_angle = self.line_follower.compute_steering_angle(frame, debug_level=self.debug_level)
                
                debug_frame = self.line_follower.get_debug_frame()
                if debug_frame is not None:
                    display_frame = debug_frame
                
                if hasattr(self.line_follower, 'current_debug_data'):
                    self.debug_data.update(self.line_follower.current_debug_data)
                
                # Apply control only if autonomous and not stopped for sign
                if self.autonomous_mode and not stopped_for_sign:
                    if current_time - self.last_frame_time >= self.frame_interval:
                        self.last_frame_time = current_time
                        self.frame_counter += 1
                        
                        self.movement_controller.apply_autonomous_control(steering_angle)
                
            except Exception as e:
                print(f"Line following error: {e}")
                self.feature_status['line_following'] = f'Runtime Error: {str(e)}'
        
        # Route debug visualization based on mode
        if self.debug_mode == "object_detection":
            display_frame = self.debug_visualizer.create_week2_debug_frame(
                display_frame, self.cache_manager, self.timing_utils, self.status_manager
            )
        elif self.debug_mode == "speed_estimation":
            display_frame = self.debug_visualizer.create_week3_debug_frame(
                display_frame, self.current_speed
            )
        # Default: line_following mode uses existing debug frame
        
        # Store frame for next speed estimation
        self.previous_frame = frame.copy()
        
        return display_frame
    
    def _run_detection_with_timing(self, frame):
        """Run object detection with timing and caching (single inference execution)"""
        return self.timing_utils.run_detection_with_timing(
            frame, self.sign_detector, self.detection_interval, 
            self.depth_interval, self.cache_manager
        )
    
    # =============================================================================
    # DEBUG MODE CONTROL
    # =============================================================================
    
    def set_debug_mode(self, mode):
        """Switch debug visualization mode"""
        if mode in self.available_modes:
            self.debug_mode = mode
            print(f"Debug mode set to: {mode}")
            return True
        else:
            print(f"❌ Invalid debug mode: {mode}. Available: {self.available_modes}")
            return False
    
    def get_debug_mode_status(self):
        """Get current debug mode and performance metrics"""
        current_time = time.time()
        return {
            'debug_mode': self.debug_mode,
            'available_modes': self.available_modes,
            'detection_fps': 1.0 / self.detection_interval if self.detection_interval > 0 else 0,
            'depth_fps': 1.0 / self.depth_interval if self.depth_interval > 0 else 0,
            'last_detection_age': current_time - self.timing_utils.last_detection_time,
            'last_depth_age': current_time - self.timing_utils.last_depth_time,
            'cached_detections': len(self.cache_manager.cached_detections),
            'last_detection_inference_ms': self.timing_utils.last_detection_inference_ms,
            'last_depth_inference_ms': self.timing_utils.last_depth_inference_ms
        }
    
    # =============================================================================
    # HARDWARE CONTROL DELEGATION
    # =============================================================================
    
    def set_camera_pan(self, angle):
        """Set camera pan angle (-90 to +90 degrees)"""
        return self.movement_controller.set_camera_pan(angle)
    
    def set_camera_tilt(self, angle):
        """Set camera tilt angle (-90 to +90 degrees)"""
        return self.movement_controller.set_camera_tilt(angle)
    
    def camera_look_down(self):
        """Preset: Point camera down for line following"""
        return self.movement_controller.camera_look_down()
    
    def camera_look_forward(self):
        """Preset: Point camera forward for obstacle detection"""
        return self.movement_controller.camera_look_forward()
    
    def move_forward(self, duration=0.5, speed=50):
        """Move robot forward for specified duration"""
        return self.movement_controller.move_forward(duration, speed, self.autonomous_mode)
    
    def move_backward(self, duration=0.5, speed=50):
        """Move robot backward for specified duration"""
        return self.movement_controller.move_backward(duration, speed, self.autonomous_mode)
    
    def turn_left(self, duration=0.5, speed=50, angle=-30):
        """Turn robot left while moving forward"""
        return self.movement_controller.turn_left(duration, speed, angle, self.autonomous_mode)
    
    def turn_right(self, duration=0.5, speed=50, angle=30):
        """Turn robot right while moving forward"""
        return self.movement_controller.turn_right(duration, speed, angle, self.autonomous_mode)
    
    def emergency_stop(self):
        """Immediately stop the robot"""
        self.autonomous_mode = False
        self.movement_controller.emergency_stop()
    
    def cleanup(self):
        """Clean shutdown of robot"""
        self.emergency_stop()
        self.movement_controller.cleanup()
    
    # =============================================================================
    # DEBUG AND CONFIGURATION METHODS
    # =============================================================================
    
    def set_debug_level(self, level):
        """Set debugging visualization level (0-4)"""
        self.debug_level = max(0, min(4, level))
        print(f"Debug level set to: {self.debug_level}")
    
    def set_frame_rate(self, fps):
        """Set target frame rate"""
        self.target_fps = max(1, min(15, fps))
        self.frame_interval = 1.0 / self.target_fps
        print(f"Frame rate set to: {self.target_fps} fps")
    
    def update_pid_parameters(self, kp=None, ki=None, kd=None):
        """Update PID parameters during runtime"""
        if self.line_follower and hasattr(self.line_follower, 'update_parameters'):
            self.line_follower.update_parameters(kp=kp, ki=ki, kd=kd)
            print(f"PID parameters updated: Kp={kp}, Ki={ki}, Kd={kd}")
        else:
            print("WARNING: Cannot update PID parameters - line follower not available")
    
    def get_debug_data(self):
        """Get clean debug data for sidebar"""
        data = self.debug_data.copy()
        
        # Add Week 2 specific data when in object detection mode
        if self.debug_mode == "object_detection":
            data.update({
                'detections_count': len(self.cache_manager.cached_detections),
                'detection_inference_ms': self.timing_utils.last_detection_inference_ms,
                'depth_inference_ms': self.timing_utils.last_depth_inference_ms,
                'stop_status': self.status_manager.get_stop_status(time.time(), self.sign_stop_until)
            })
        
        return data
    
    def get_feature_status(self):
        """Return current status of all features"""
        return {
            'autonomous_mode': self.autonomous_mode,
            'features': self.feature_status.copy(),
            'camera_position': self.movement_controller.get_camera_position(),
            'target_fps': self.target_fps,
            'debug_level': self.debug_level,
            'debug_mode': self.debug_mode
        }

# Global robot instance
robot = RobotController()