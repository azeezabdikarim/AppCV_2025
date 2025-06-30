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
            self.last_detection_time = 0
            self.last_depth_time = 0
            self.last_detection_inference_ms = 0
            self.last_depth_inference_ms = 0
            
            # Cached results for debugging
            self.cached_detections = []
            self.cached_depth_map = None
            self.detection_frame_counter = 0
            self.depth_frame_counter = 0
            
            # Anti-infinite-stop logic
            self.recently_stopped_for_sign = False
            self.stop_cooldown_until = None
            
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
        
        # Check if currently stopped for sign detection and cooldown logic
        current_time = time.time()
        stopped_for_sign = (self.sign_stop_until is not None and current_time < self.sign_stop_until)
        in_cooldown = (self.stop_cooldown_until is not None and current_time < self.stop_cooldown_until)
        
        # Week 2: Sign Detection (with timing and caching)
        if self.sign_detector and FEATURES_ENABLED['sign_detection'] and not stopped_for_sign and not in_cooldown:
            detected_signs = self._run_detection_with_timing(frame)
            if self.sign_detector.should_stop(detected_signs, frame):
                self.sign_stop_until = current_time + self.sign_stop_duration
                self.recently_stopped_for_sign = True
                stopped_for_sign = True
                print("🛑 Stopping for detected sign")
        
        # Anti-infinite-stop: Reset cooldown when robot starts moving again
        if self.recently_stopped_for_sign and not stopped_for_sign and self.autonomous_mode:
            self.stop_cooldown_until = current_time + self.stop_cooldown_duration
            self.recently_stopped_for_sign = False
            print("🔄 Stop cooldown activated")
        
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
                            self.picar.forward(100)
                
            except Exception as e:
                print(f"Line following error: {e}")
                self.feature_status['line_following'] = f'Runtime Error: {str(e)}'
        
        # Route debug visualization based on mode
        if self.debug_mode == "object_detection":
            display_frame = self._create_week2_debug_frame(display_frame)
        elif self.debug_mode == "speed_estimation":
            # Week 3 debug frame (placeholder for now)
            display_frame = self._create_week3_debug_frame(display_frame)
        # Default: line_following mode uses existing debug frame
        
        # Store frame for next speed estimation
        self.previous_frame = frame.copy()
        
        return display_frame
    
    # =============================================================================
    # WEEK 2 DEBUG SYSTEM - OBJECT DETECTION & DEPTH ANALYSIS
    # =============================================================================
    
    def _run_detection_with_timing(self, frame):
        """Run object detection with timing and caching (single inference execution)"""
        current_time = time.time()
        
        # Only run detection if enough time has passed
        if current_time - self.last_detection_time >= self.detection_interval:
            start_time = time.perf_counter()
            detected_signs = self.sign_detector.detect_signs(frame)
            self.last_detection_inference_ms = (time.perf_counter() - start_time) * 1000
            
            # Cache results for debugging
            self.cached_detections = detected_signs
            self.last_detection_time = current_time
            self.detection_frame_counter += 1
            
            # Run depth analysis if we have detections and it's time
            if detected_signs and self._should_run_depth():
                self._run_depth_with_timing(frame, detected_signs)
            
            return detected_signs
        else:
            # Return cached results - no additional inference
            return self.cached_detections
    
    def _should_run_depth(self):
        """Check if it's time to run depth analysis"""
        current_time = time.time()
        return current_time - self.last_depth_time >= self.depth_interval
    
    def _run_depth_with_timing(self, frame, detections):
        """Run depth estimation with timing (placeholder for MiDaS integration)"""
        start_time = time.perf_counter()
        
        # Placeholder: Simple area-based "depth" for now
        # Students will replace this with actual MiDaS integration
        depth_map = self._simple_area_based_depth(frame, detections)
        
        self.last_depth_inference_ms = (time.perf_counter() - start_time) * 1000
        self.cached_depth_map = depth_map
        self.last_depth_time = time.time()
        self.depth_frame_counter = self.detection_frame_counter
    
    def _simple_area_based_depth(self, frame, detections):
        """Simple area-based depth estimation (students will replace with MiDaS)"""
        # Create a simple depth visualization based on bounding box areas
        height, width = frame.shape[:2]
        depth_map = np.zeros((height, width), dtype=np.uint8)
        
        for detection in detections:
            bbox = detection['bbox']
            x, y, w, h = bbox
            area = w * h
            
            # Larger bounding box = closer = higher depth value
            depth_value = min(255, int(area / 100))  # Simple area-to-depth conversion
            cv2.rectangle(depth_map, (x, y), (x + w, y + h), depth_value, -1)
        
        return depth_map
    
    def _create_week2_debug_frame(self, original_frame):
        """Create dual-panel visualization for Week 2 object detection and depth"""
        height, width = original_frame.shape[:2]
        
        # Left panel: Object detection overlay
        left_panel = original_frame.copy()
        if self.cached_detections:
            left_panel = self._draw_detection_overlay(left_panel, self.cached_detections)
        
        # Add detection panel label
        cv2.putText(left_panel, "Object Detection", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Right panel: Depth analysis
        right_panel = original_frame.copy()
        if self.cached_depth_map is not None:
            # Apply depth colormap
            depth_colored = cv2.applyColorMap(self.cached_depth_map, cv2.COLORMAP_PLASMA)
            right_panel = cv2.addWeighted(right_panel, 0.6, depth_colored, 0.4, 0)
            
            # Show frame counter alignment
            cv2.putText(right_panel, "[0]", (width - 40, height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        else:
            # Show waiting status with frame delay counter
            delay = self.detection_frame_counter - self.depth_frame_counter
            if delay > 0:
                cv2.putText(right_panel, f"[-{delay}]", (width - 50, height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Add depth panel label
        cv2.putText(right_panel, "Depth Analysis", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Combine panels side by side
        combined = np.hstack([left_panel, right_panel])
        
        # Add comprehensive debug information
        return self._add_week2_debug_overlay(combined)
    
    def _draw_detection_overlay(self, frame, detections):
        """Draw bounding boxes and labels for detected objects"""
        for detection in detections:
            bbox = detection['bbox']
            x, y, w, h = bbox
            confidence = detection['confidence']
            class_name = detection.get('class_name', detection.get('class', 'object'))
            
            # Color coding for different object types
            if 'stop' in class_name.lower():
                color = (0, 0, 255)  # Red for stop signs
            else:
                color = (0, 255, 0)  # Green for other objects
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with confidence
            label = f"{class_name}: {confidence:.2f}"
            font_scale = 0.5
            thickness = 1
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            # Background for label
            cv2.rectangle(frame, (x, y - label_size[1] - 5), 
                         (x + label_size[0], y), color, -1)
            
            # Label text
            cv2.putText(frame, label, (x, y - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def _add_week2_debug_overlay(self, combined_frame):
        """Add comprehensive Week 2 debug information"""
        height, width = combined_frame.shape[:2]
        
        # Performance metrics
        detection_fps = 1.0 / self.detection_interval if self.detection_interval > 0 else 0
        depth_fps = 1.0 / self.depth_interval if self.depth_interval > 0 else 0
        
        current_time = time.time()
        stopped_for_sign = (self.sign_stop_until is not None and current_time < self.sign_stop_until)
        in_cooldown = (self.stop_cooldown_until is not None and current_time < self.stop_cooldown_until)
        
        if stopped_for_sign:
            status = "STOPPED"
        elif in_cooldown:
            status = "COOLDOWN"
        else:
            status = "ACTIVE"
        
        # Top status line
        status_text = f"Status: {status} | Detection: {detection_fps:.1f}fps ({self.last_detection_inference_ms:.0f}ms) | Depth: {depth_fps:.1f}fps ({self.last_depth_inference_ms:.0f}ms)"
        cv2.putText(combined_frame, status_text, (10, height - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Detection list
        detection_text = f"Detections: {len(self.cached_detections)}"
        if self.cached_detections:
            top_detections = self.cached_detections[:3]  # Show top 3
            detection_names = [f"{d.get('class_name', d.get('class', 'obj'))}({d['confidence']:.2f})" 
                             for d in top_detections]
            detection_text += f" - {', '.join(detection_names)}"
        
        cv2.putText(combined_frame, detection_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return combined_frame
    
    def _create_week3_debug_frame(self, original_frame):
        """Placeholder for Week 3 speed estimation debug frame"""
        # Add simple speed display for now
        speed_text = f"Speed: {self.current_speed:.1f} units/s"
        cv2.putText(original_frame, speed_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(original_frame, "Speed Estimation Mode (Week 3)", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return original_frame
    
    # =============================================================================
    # DEBUG MODE CONTROL
    # =============================================================================
    
    def set_debug_mode(self, mode):
        """Switch debug visualization mode"""
        if mode in self.available_modes:
            self.debug_mode = mode
            print(f"🔧 Debug mode set to: {mode}")
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
            'last_detection_age': current_time - self.last_detection_time,
            'last_depth_age': current_time - self.last_depth_time,
            'cached_detections': len(self.cached_detections),
            'last_detection_inference_ms': self.last_detection_inference_ms,
            'last_depth_inference_ms': self.last_depth_inference_ms
        }
    
    # =============================================================================
    # CAMERA CONTROL METHODS (unchanged)
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
        data = self.debug_data.copy()
        
        # Add Week 2 specific data when in object detection mode
        if self.debug_mode == "object_detection":
            data.update({
                'detections_count': len(self.cached_detections),
                'detection_inference_ms': self.last_detection_inference_ms,
                'depth_inference_ms': self.last_depth_inference_ms,
                'stop_status': self._get_stop_status()
            })
        
        return data
    
    def _get_stop_status(self):
        """Get current stopping status"""
        current_time = time.time()
        if self.sign_stop_until is not None and current_time < self.sign_stop_until:
            return "STOPPED"
        elif self.stop_cooldown_until is not None and current_time < self.stop_cooldown_until:
            return "COOLDOWN"
        else:
            return "ACTIVE"
    
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
            'debug_level': self.debug_level,
            'debug_mode': self.debug_mode
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