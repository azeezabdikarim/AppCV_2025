#!/usr/bin/env python3

import cv2
import numpy as np
import time

class DebugVisualizer:
    """Handles all debug visualization and overlay creation"""
    
    def __init__(self):
        """Initialize the debug visualizer"""
        pass
    
    def create_week2_debug_frame(self, original_frame, cache_manager, timing_utils, status_manager):
        """Create dual-panel visualization for Week 2 object detection and depth"""
        height, width = original_frame.shape[:2]
        
        # Left panel: Object detection overlay
        left_panel = original_frame.copy()
        if cache_manager.cached_detections:
            left_panel = self._draw_detection_overlay(left_panel, cache_manager.cached_detections)
        
        # Add detection panel label
        cv2.putText(left_panel, "Object Detection", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Right panel: Depth analysis
        right_panel = original_frame.copy()
        if cache_manager.cached_depth_map is not None:
            # Apply depth colormap
            depth_colored = cv2.applyColorMap(cache_manager.cached_depth_map, cv2.COLORMAP_PLASMA)
            right_panel = cv2.addWeighted(right_panel, 0.6, depth_colored, 0.4, 0)
            
            # Show frame counter alignment
            cv2.putText(right_panel, "[0]", (width - 40, height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        else:
            # Show waiting status with frame delay counter
            delay = cache_manager.detection_frame_counter - cache_manager.depth_frame_counter
            if delay > 0:
                cv2.putText(right_panel, f"[-{delay}]", (width - 50, height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Add depth panel label
        cv2.putText(right_panel, "Depth Analysis", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Combine panels side by side
        combined = np.hstack([left_panel, right_panel])
        
        # Add comprehensive debug information
        return self._add_week2_debug_overlay(combined, cache_manager, timing_utils, status_manager)
    
    def create_week3_debug_frame(self, original_frame, current_speed):
        """Placeholder for Week 3 speed estimation debug frame"""
        # Add simple speed display for now
        speed_text = f"Speed: {current_speed:.1f} units/s"
        cv2.putText(original_frame, speed_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(original_frame, "Speed Estimation Mode (Week 3)", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return original_frame
    
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
    
    def _add_week2_debug_overlay(self, combined_frame, cache_manager, timing_utils, status_manager):
        """Add comprehensive Week 2 debug information"""
        height, width = combined_frame.shape[:2]
        
        # Performance metrics
        detection_interval = getattr(timing_utils, 'detection_interval', 0.5)
        depth_interval = getattr(timing_utils, 'depth_interval', 1.0)
        
        detection_fps = 1.0 / detection_interval if detection_interval > 0 else 0
        depth_fps = 1.0 / depth_interval if depth_interval > 0 else 0
        
        current_time = time.time()
        
        # Get status from status manager
        stop_status = status_manager.get_current_status(current_time)
        
        # Top status line
        status_text = f"Status: {stop_status} | Detection: {detection_fps:.1f}fps ({timing_utils.last_detection_inference_ms:.0f}ms) | Depth: {depth_fps:.1f}fps ({timing_utils.last_depth_inference_ms:.0f}ms)"
        cv2.putText(combined_frame, status_text, (10, height - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Detection list
        detection_text = f"Detections: {len(cache_manager.cached_detections)}"
        if cache_manager.cached_detections:
            top_detections = cache_manager.cached_detections[:3]  # Show top 3
            detection_names = [f"{d.get('class_name', d.get('class', 'obj'))}({d['confidence']:.2f})" 
                             for d in top_detections]
            detection_text += f" - {', '.join(detection_names)}"
        
        cv2.putText(combined_frame, detection_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return combined_frame