#!/usr/bin/env python3

import threading
import time
import cv2
import numpy as np
import sys
import os

# Add the parent directory to Python path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from picarx import Picarx
except ImportError:
    print("⚠️  PicarX not available - running in simulation mode")
    Picarx = None

# =============================================================================
# EXPLICIT FEATURE CONTROL (Option A)
# Students enable features when they're ready
# =============================================================================
FEATURES_ENABLED = {
    'line_following': True,   # Week 1 - Enable when ready
    'sign_detection': False,  # Week 2 - Student enables when implemented  
    'speed_estimation': False # Week 3 - Student enables when implemented
}

class RobotController:
    def __init__(self):
        """Initialize the robot controller"""
        try:
            if Picarx:
                self.picar = Picarx()
                print("✅ PiCar-X hardware connected")
            else:
                self.picar = None
                print("⚠️  Running without PiCar-X hardware")
                
            self.is_moving = False
            
            # Autonomous mode variables
            self.autonomous_mode = False
            self.frame_counter = 0
            self.previous_frame = None
            self.current_speed = 0.0
            self.sign_stop_until = None      # Time when sign stop expires
            self.sign_stop_duration = 3.0    # Seconds to stop for detected signs
            
            # Debug and performance settings
            self.debug_level = 0
            self.target_fps = 10
            self.last_frame_time = time.time()
            self.frame_interval = 1.0 / self.target_fps
            
            # Camera positioning
            self.camera_pan_angle = 0   # -90 to +90 degrees
            self.camera_tilt_angle = -30  # Start looking down for line following
            
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
            
            # Initialize camera position
            self._set_camera_position()
            
            # Load enabled features
            self._load_enabled_features()
            
            print("Robot controller initialized successfully")
        except Exception as e:
            print(f"Error initializing robot: {e}")
            self.picar = None
    
    def _set_camera_position(self):
        """Set camera to initial position"""
        if self.picar:
            try:
                self.picar.set_cam_pan_angle(self.camera_pan_angle)
                self.picar.set_cam_tilt_angle(self.camera_tilt_angle)
                print(f"📷 Camera positioned: pan={self.camera_pan_angle}°, tilt={self.camera_tilt_angle}°")
            except Exception as e:
                print(f"⚠️  Camera positioning error: {e}")
    
    def _load_enabled_features(self):
        """Load only the features that are explicitly enabled"""
        
        print("🔍 Loading enabled features...")
        
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
            print("⚪ Line following disabled")
            
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
            print("⚪ Sign detection disabled")
            
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
            print("⚪ Speed estimation disabled")
        
        # Print final status
        print("📊 Feature Status Summary:")
        for feature, status in self.feature_status.items():
            print(f"   {feature}: {status}")
    
    def start_autonomous_mode(self):
        """Start autonomous line following mode"""
        print("🚀 Starting autonomous mode...")
        
        if not FEATURES_ENABLED['line_following']:
            error_msg = "Line following is disabled in feature config"
            print(f"❌ {error_msg}")
            return False, error_msg
        
        if not self.line_follower:
            error_msg = f"Line following not available: {self.feature_status['line_following']}"
            print(f"❌ {error_msg}")
            return False, error_msg
            
        if not self.picar:
            error_msg = "Robot hardware not connected"
            print(f"❌ {error_msg}")
            return False, error_msg
            
        self.autonomous_mode = True
        self.frame_counter = 0
        self.debug_data['mode'] = 'Autonomous'
        print("🤖 Autonomous mode started successfully")
        return True, "Autonomous mode started"
    
    def stop_autonomous_mode(self):
        """Stop autonomous mode and return to manual control"""
        self.autonomous_mode = False
        self.debug_data['mode'] = 'Manual'
        if self.picar:
            self.picar.stop()
            self.picar.set_dir_servo_angle(0)
        print("🛑 Autonomous mode stopped")
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
        
        # Check if currently stopped for sign detection
        current_time = time.time()
        stopped_for_sign = (self.sign_stop_until is not None and current_time < self.sign_stop_until)
        
        # Week 2: Sign Detection
        if self.sign_detector and FEATURES_ENABLED['sign_detection'] and not stopped_for_sign:
            detected_signs = self.sign_detector.detect_signs(frame)
            if self.sign_detector.should_stop(detected_signs, frame):
                self.sign_stop_until = current_time + self.sign_stop_duration
                stopped_for_sign = True
                print("🛑 Stopping for detected sign")
        
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
                        
                        if self.picar:
                            self.picar.set_dir_servo_angle(steering_angle)
<<<<<<< HEAD
                            self.picar.forward(100)
=======
                            self.picar.forward(10)  # 10% speed
>>>>>>> refs/remotes/origin/main
                
            except Exception as e:
                print(f"Line following error: {e}")
                self.feature_status['line_following'] = f'Runtime Error: {str(e)}'
        
        # Store frame for next speed estimation
        self.previous_frame = frame.copy()
        
        return display_frame
    
    # =============================================================================
    # CAMERA CONTROL METHODS
    # =============================================================================
    
    def set_camera_pan(self, angle):
        """Set camera pan angle (-90 to +90 degrees)"""
        angle = max(-90, min(90, angle))
        self.camera_pan_angle = angle
        
        if self.picar:
            try:
                self.picar.set_cam_pan_angle(angle)
                print(f"📷 Camera pan set to {angle}°")
                return True
            except Exception as e:
                print(f"❌ Camera pan error: {e}")
                return False
        return False
    
    def set_camera_tilt(self, angle):
        """Set camera tilt angle (-90 to +90 degrees)"""
        angle = max(-90, min(90, angle))
        self.camera_tilt_angle = angle
        
        if self.picar:
            try:
                self.picar.set_cam_tilt_angle(angle)
                print(f"📷 Camera tilt set to {angle}°")
                return True
            except Exception as e:
                print(f"❌ Camera tilt error: {e}")
                return False
        return False
    
    def camera_look_down(self):
        """Preset: Point camera down for line following"""
        return self.set_camera_pan(0) and self.set_camera_tilt(-30)
    
    def camera_look_forward(self):
        """Preset: Point camera forward for obstacle detection"""
        return self.set_camera_pan(0) and self.set_camera_tilt(0)
    
    # =============================================================================
    # DEBUG AND CONFIGURATION METHODS
    # =============================================================================
    
    def set_debug_level(self, level):
        """Set debugging visualization level (0-4)"""
        self.debug_level = max(0, min(4, level))
        print(f"🔧 Debug level set to: {self.debug_level}")
    
    def set_frame_rate(self, fps):
        """Set target frame rate"""
        self.target_fps = max(1, min(15, fps))
        self.frame_interval = 1.0 / self.target_fps
        print(f"🔧 Frame rate set to: {self.target_fps} fps")
    
    def update_pid_parameters(self, kp=None, ki=None, kd=None):
        """Update PID parameters during runtime"""
        if self.line_follower and hasattr(self.line_follower, 'update_parameters'):
            self.line_follower.update_parameters(kp=kp, ki=ki, kd=kd)
            print(f"🔧 PID parameters updated: Kp={kp}, Ki={ki}, Kd={kd}")
        else:
            print("⚠️  Cannot update PID parameters - line follower not available")
    
    def get_debug_data(self):
        """Get clean debug data for sidebar"""
        return self.debug_data.copy()
    
    def get_feature_status(self):
        """Return current status of all features"""
        return {
            'autonomous_mode': self.autonomous_mode,
            'features': self.feature_status.copy(),
            'camera_position': {
                'pan': self.camera_pan_angle,
                'tilt': self.camera_tilt_angle
            },
            'target_fps': self.target_fps,
            'debug_level': self.debug_level
        }
    
    # =============================================================================
    # MANUAL CONTROL METHODS (unchanged)
    # =============================================================================
    
    def _auto_stop(self):
        """Automatically stop the robot and center wheels after movement"""
        if self.picar:
            self.picar.stop()
            self.picar.set_dir_servo_angle(0)
            self.is_moving = False
    
    def move_forward(self, duration=0.5, speed=50):
        """Move robot forward for specified duration"""
        if not self.picar or self.is_moving or self.autonomous_mode:
            return False
        
        try:
            self.is_moving = True
            self.picar.set_dir_servo_angle(0)
            self.picar.forward(speed)
            timer = threading.Timer(duration, self._auto_stop)
            timer.start()
            return True
        except Exception as e:
            print(f"Error moving forward: {e}")
            self._auto_stop()
            return False
    
    def move_backward(self, duration=0.5, speed=50):
        """Move robot backward for specified duration"""
        if not self.picar or self.is_moving or self.autonomous_mode:
            return False
        
        try:
            self.is_moving = True
            self.picar.set_dir_servo_angle(0)
            self.picar.backward(speed)
            timer = threading.Timer(duration, self._auto_stop)
            timer.start()
            return True
        except Exception as e:
            print(f"Error moving backward: {e}")
            self._auto_stop()
            return False
    
    def turn_left(self, duration=0.5, speed=50, angle=-30):
        """Turn robot left while moving forward"""
        if not self.picar or self.is_moving or self.autonomous_mode:
            return False
        
        try:
            self.is_moving = True
            self.picar.set_dir_servo_angle(angle)
            self.picar.forward(speed)
            timer = threading.Timer(duration, self._auto_stop)
            timer.start()
            return True
        except Exception as e:
            print(f"Error turning left: {e}")
            self._auto_stop()
            return False
    
    def turn_right(self, duration=0.5, speed=50, angle=30):
        """Turn robot right while moving forward"""
        if not self.picar or self.is_moving or self.autonomous_mode:
            return False
        
        try:
            self.is_moving = True
            self.picar.set_dir_servo_angle(angle)
            self.picar.forward(speed)
            timer = threading.Timer(duration, self._auto_stop)
            timer.start()
            return True
        except Exception as e:
            print(f"Error turning right: {e}")
            self._auto_stop()
            return False
    
    def emergency_stop(self):
        """Immediately stop the robot"""
        self.autonomous_mode = False
        if self.picar:
            self.picar.stop()
            self.picar.set_dir_servo_angle(0)
            self.is_moving = False
        print("Emergency stop activated")
    
    def cleanup(self):
        """Clean shutdown of robot"""
        if self.picar:
            self.emergency_stop()
        print("Robot controller cleaned up")

# Global robot instance
robot = RobotController()
