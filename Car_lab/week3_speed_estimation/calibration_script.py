#!/usr/bin/env python3
"""
Week 3 Speed Estimation - Calibration Data Collection Script
===========================================================

This script automatically collects optical flow calibration data by:
1. Recording video while driving the robot at specified power for a known distance
2. Processing optical flow calculations offline from the recorded video
3. Calculating the relationship between optical flow and real speed
4. Saving results to cumulative CSV file

Usage:
    python calibration_script.py --dist 2.0 --pow 30
    python calibration_script.py --dist 2.5 --pow 40
    python calibration_script.py --dist 3.0 --pow 50
"""

import argparse
import time
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import sys

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import robot hardware modules
from core.camera_manager import camera
from core.utils.movement_controls import MovementController

class CalibrationCollector:
    def __init__(self):
        """Initialize calibration data collector"""
        
        # File paths
        self.csv_file = Path("calibration_data.csv")
        self.videos_dir = Path("videos")
        self.videos_dir.mkdir(exist_ok=True)
        
        # Hardware setup
        self.movement_controller = MovementController()
        
        # Ultrasonic sensor setup
        self.ultrasonic_stop_distance = 0.2  # meters from wall
        
        # Optical flow parameters
        self.feature_params = {
            'maxCorners': 100,
            'qualityLevel': 0.2,
            'minDistance': 10,
            'blockSize': 7
        }
        
        self.lk_params = {
            'winSize': (15, 15),
            'maxLevel': 2,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }
        
        # Ensure CSV exists with headers
        self._initialize_csv()
        
        print("✅ Calibration collector initialized")
        print(f"📁 Data will be saved to: {self.csv_file}")
        print(f"🎥 Videos will be saved to: {self.videos_dir}/")

    def _initialize_csv(self):
        """Create CSV file with headers if it doesn't exist"""
        if not self.csv_file.exists():
            headers = [
                'run_id', 'timestamp', 'distance', 'motor_power', 
                'recorded_time', 'calculated_speed', 'avg_optical_flow', 
                'num_features', 'include_in_calibration',
                'cumulative_slope', 'cumulative_intercept', 'cumulative_r_squared', 'cumulative_n_points'
            ]
            
            df = pd.DataFrame(columns=headers)
            df.to_csv(self.csv_file, index=False)
            print(f"📝 Created new calibration data file: {self.csv_file}")
        else:
            print(f"📖 Using existing calibration data file: {self.csv_file}")

    def get_ultrasonic_distance(self):
        """Get distance from PiCar-X ultrasonic sensor"""
        try:
            # Using PiCar-X ultrasonic sensor through movement controller
            distance = self.movement_controller.picar.ultrasonic.read()
            return distance / 100.0  # Convert cm to meters
        except Exception as e:
            print(f"Ultrasonic sensor error: {e}")
            return 1.0  # Default safe distance

    def wait_for_camera_ready(self):
        """Wait for camera to be properly initialized and streaming with real frames"""
        print("📷 Waiting for camera to be ready...")
        
        # First, explicitly start streaming if not already running
        if not camera.is_running:
            print("   Starting camera streaming...")
            camera.start_streaming()
            time.sleep(2)  # Give camera time to initialize
        
        # Wait for actual camera frames (not placeholder frames)
        max_attempts = 15
        for attempt in range(max_attempts):
            try:
                frame = camera.get_frame()
                if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                    # Check if this is a real camera frame, not a placeholder
                    # Placeholder frames are solid color or have text - real frames have more variation
                    
                    # Convert to grayscale to check variance
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    variance = np.var(gray)
                    
                    # Real camera frames should have significant pixel variance
                    # Placeholder frames typically have very low variance
                    if variance > 100:  # Threshold for real vs placeholder frames
                        print("✅ Camera ready with real frames")
                        print(f"   Frame variance: {variance:.1f} (good)")
                        return True
                    else:
                        print(f"   Attempt {attempt + 1}/{max_attempts} - placeholder frame detected (variance: {variance:.1f})")
                else:
                    print(f"   Attempt {attempt + 1}/{max_attempts} - no frame received")
            except Exception as e:
                print(f"   Attempt {attempt + 1}/{max_attempts} - error: {e}")
            
            time.sleep(1)
        
        print("❌ Camera failed to provide real frames")
        print("   Still getting placeholder frames after initialization")
        return False

    def record_video_during_movement(self, distance, motor_power):
        """Record video while robot moves, return video path and movement stats"""
        
        # Get next run ID
        if self.csv_file.exists():
            df = pd.read_csv(self.csv_file)
            run_id = len(df) + 1
        else:
            run_id = 1
        
        # Setup video recording
        video_filename = f"run_{run_id:03d}_{distance:.1f}m_{motor_power}pct_raw.mp4"
        video_path = self.videos_dir / video_filename
        
        # Make sure camera is ready
        if not self.wait_for_camera_ready():
            print("❌ Cannot proceed without camera")
            return None, None
        
        # Setup for data collection
        print(f"🎯 Position robot {distance}m from wall")
        print(f"🛑 Robot will auto-stop at {self.ultrasonic_stop_distance}m from wall")
        input("👆 Press Enter when robot is positioned correctly...")
        
        # Verify initial distance
        initial_distance = self.get_ultrasonic_distance()
        print(f"📡 Initial distance reading: {initial_distance:.2f}m")
        
        if abs(initial_distance - distance) > 0.5:
            print(f"⚠️  Warning: Expected {distance}m but reading {initial_distance:.2f}m")
            print("📝 Continuing with actual distance reading...")
        
        # Get first frame to setup video writer
        first_frame = camera.get_frame()
        if first_frame is None:
            print("❌ Cannot get camera frame")
            return None, None
            
        height, width = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 20
        
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        print(f"🎬 Starting video recording...")
        print(f"🚀 Starting motor at {motor_power}% power...")
        
        # Start robot movement
        self.movement_controller.picar.set_dir_servo_angle(0)  # Straight
        self.movement_controller.picar.forward(motor_power)
        self.movement_controller.picar.set_cam_tilt_angle(-10)
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                # Get current frame
                current_frame = camera.get_frame()
                if current_frame is not None:
                    # Record raw frame without any text overlays
                    out.write(current_frame)
                    frame_count += 1
                
                # Check if we should stop (ultrasonic sensor)
                current_distance = self.get_ultrasonic_distance()
                if current_distance <= self.ultrasonic_stop_distance:
                    print(f"🛑 Stopping - reached {current_distance:.2f}m from wall")
                    break
                
                # Safety timeout
                current_time = time.time() - start_time
                if current_time > 15.0:  # 15 second max run time
                    print("⏰ Timeout - stopping for safety")
                    break
                
                time.sleep(0.05)  # 20 FPS recording
                
        finally:
            # Stop the robot and close video
            self.movement_controller.picar.stop()
            out.release()
            print("✋ Robot stopped")
        
        # Calculate movement results
        total_time = time.time() - start_time
        actual_distance = distance - self.ultrasonic_stop_distance
        calculated_speed = actual_distance / total_time
        
        movement_stats = {
            'run_id': run_id,
            'total_time': total_time,
            'actual_distance': actual_distance,
            'calculated_speed': calculated_speed,
            'frame_count': frame_count
        }
        
        print(f"📊 MOVEMENT RESULTS:")
        print(f"   Time: {total_time:.2f}s")
        print(f"   Distance: {actual_distance:.2f}m")
        print(f"   Speed: {calculated_speed:.3f} m/s")
        print(f"   Frames recorded: {frame_count}")
        print(f"🎥 Raw video saved: {video_path}")
        
        return video_path, movement_stats

    def calculate_optical_flow(self, prev_frame, curr_frame):
        """Calculate optical flow magnitude between two frames"""
        
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features in previous frame
        features = cv2.goodFeaturesToTrack(prev_gray, **self.feature_params)
        
        if features is None or len(features) == 0:
            return 0.0, 0, []
        
        # Track features using Lucas-Kanade
        new_features, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, features, None, **self.lk_params)
        
        # Calculate flow magnitudes for successfully tracked features
        flow_magnitudes = []
        good_features = []
        
        for i, (prev_pt, curr_pt) in enumerate(zip(features, new_features)):
            if status[i] == 1:  # Successfully tracked
                dx = curr_pt[0][0] - prev_pt[0][0]
                dy = curr_pt[0][1] - prev_pt[0][1]
                magnitude = np.sqrt(dx*dx + dy*dy)
                flow_magnitudes.append(magnitude)
                good_features.append((prev_pt, curr_pt, magnitude))
        
        avg_flow = np.mean(flow_magnitudes) if flow_magnitudes else 0.0
        return avg_flow, len(good_features), good_features

    def create_flow_visualization(self, frame, features_data):
        """Create visualization of optical flow on frame"""
        vis_frame = frame.copy()
        
        # Draw tracked features and flow vectors
        for prev_pt, curr_pt, magnitude in features_data:
            # Draw feature points
            cv2.circle(vis_frame, tuple(prev_pt[0].astype(int)), 3, (0, 255, 0), -1)
            cv2.circle(vis_frame, tuple(curr_pt[0].astype(int)), 3, (0, 0, 255), -1)
            
            # Draw flow vector
            cv2.arrowedLine(vis_frame, 
                          tuple(prev_pt[0].astype(int)), 
                          tuple(curr_pt[0].astype(int)), 
                          (255, 0, 255), 2, tipLength=0.3)
        
        return vis_frame

    def process_optical_flow_from_video(self, video_path, movement_stats):
        """Process optical flow from recorded video"""
        
        print(f"🔄 Processing optical flow from video...")
        
        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("❌ Could not open video file")
            return None
        
        # Setup output video with flow visualization
        flow_video_path = video_path.parent / video_path.name.replace('_raw.mp4', '_flow.mp4')
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(flow_video_path), fourcc, fps, (width, height))
        
        # Process frames
        flow_data = []
        prev_frame = None
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            
            if prev_frame is not None:
                # Calculate optical flow
                flow_mag, num_features, features_list = self.calculate_optical_flow(prev_frame, frame)
                
                if flow_mag > 0:
                    flow_data.append({
                        'flow_magnitude': flow_mag,
                        'num_features': num_features,
                        'frame_number': frame_number
                    })
                
                # Create visualization
                vis_frame = self.create_flow_visualization(frame, features_list)
                
                # Add flow info to frame
                cv2.putText(vis_frame, f"Flow: {flow_mag:.1f} px/frame", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Features: {num_features}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                out.write(vis_frame)
            else:
                # First frame - just write as is
                out.write(frame)
            
            prev_frame = frame
        
        cap.release()
        out.release()
        
        # Calculate average optical flow
        if flow_data:
            avg_optical_flow = np.mean([f['flow_magnitude'] for f in flow_data])
            avg_features = np.mean([f['num_features'] for f in flow_data])
        else:
            avg_optical_flow = 0.0
            avg_features = 0
            print("⚠️  No optical flow data calculated!")
        
        print(f"✅ Optical flow processing complete")
        print(f"   Average flow: {avg_optical_flow:.2f} px/frame")
        print(f"   Average features: {avg_features:.0f}")
        print(f"🎥 Flow visualization saved: {flow_video_path}")
        
        return {
            'avg_optical_flow': avg_optical_flow,
            'num_features': avg_features,
            'flow_video_path': flow_video_path
        }

    def run_calibration(self, distance, motor_power):
        """Run a single calibration data collection sequence"""
        
        print(f"\n{'='*50}")
        print(f"🚗 CALIBRATION RUN")
        print(f"📏 Target distance: {distance}m")
        print(f"⚡ Motor power: {motor_power}%")
        print(f"{'='*50}")
        
        # Step 1: Record video during movement
        video_path, movement_stats = self.record_video_during_movement(distance, motor_power)
        
        if video_path is None or movement_stats is None:
            print("❌ Video recording failed")
            return
        
        # Step 2: Process optical flow from recorded video
        flow_results = self.process_optical_flow_from_video(video_path, movement_stats)
        
        if flow_results is None:
            print("❌ Optical flow processing failed")
            return
        
        # Step 3: Save results to CSV
        self._save_to_csv(
            movement_stats['run_id'],
            distance,
            motor_power,
            movement_stats['total_time'],
            movement_stats['calculated_speed'],
            flow_results['avg_optical_flow'],
            flow_results['num_features']
        )
        
        print(f"💾 Data saved to CSV")

    def _save_to_csv(self, run_id, distance, motor_power, recorded_time, 
                     calculated_speed, avg_optical_flow, num_features):
        """Save run data to CSV and calculate cumulative statistics"""
        
        # Create new row data
        new_data = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'distance': distance,
            'motor_power': motor_power,
            'recorded_time': recorded_time,
            'calculated_speed': calculated_speed,
            'avg_optical_flow': avg_optical_flow,
            'num_features': num_features,
            'include_in_calibration': True
        }
        
        # Load existing data
        if self.csv_file.exists():
            df = pd.read_csv(self.csv_file)
        else:
            df = pd.DataFrame()
        
        # Add new row
        new_row = pd.DataFrame([new_data])
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Calculate cumulative statistics
        calibration_data = df[df['include_in_calibration'] == True]
        
        if len(calibration_data) >= 2:
            flows = calibration_data['avg_optical_flow'].values
            speeds = calibration_data['calculated_speed'].values
            
            # Linear regression: speed = slope * flow + intercept
            slope, intercept = np.polyfit(flows, speeds, 1)
            
            # Calculate R-squared
            predicted_speeds = slope * flows + intercept
            ss_res = np.sum((speeds - predicted_speeds) ** 2)
            ss_tot = np.sum((speeds - np.mean(speeds)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            n_points = len(calibration_data)
        else:
            slope, intercept, r_squared, n_points = 0, 0, 0, len(calibration_data)
        
        # Update the last row with cumulative stats
        df.loc[df.index[-1], 'cumulative_slope'] = slope
        df.loc[df.index[-1], 'cumulative_intercept'] = intercept
        df.loc[df.index[-1], 'cumulative_r_squared'] = r_squared
        df.loc[df.index[-1], 'cumulative_n_points'] = n_points
        
        # Save CSV
        df.to_csv(self.csv_file, index=False)
        
        # Print cumulative results
        print(f"\n🔄 CUMULATIVE CALIBRATION ({n_points} runs):")
        if n_points >= 2:
            print(f"   Linear fit: speed = {slope:.6f} × flow + ({intercept:.6f})")
            print(f"   R-squared: {r_squared:.3f}")
            print(f"\n📋 Copy to speed_estimator.py:")
            print(f"   self.flow_to_speed_slope = {slope:.6f}")
            print(f"   self.flow_to_speed_intercept = {intercept:.6f}")
        else:
            print(f"   Need at least 2 runs for calibration")


def main():
    parser = argparse.ArgumentParser(description='Collect optical flow calibration data')
    parser.add_argument('--dist', type=float, required=True,
                       help='Distance from wall in meters (e.g., 2.0)')
    parser.add_argument('--pow', type=int, required=True,
                       help='Motor power percentage (e.g., 30)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.dist <= 0 or args.dist > 5:
        print("❌ Error: Distance must be between 0 and 5 meters")
        return
    
    if args.pow <= 0 or args.pow > 100:
        print("❌ Error: Power must be between 1 and 100 percent")
        return
    
    # Run calibration
    collector = CalibrationCollector()
    collector.run_calibration(args.dist, args.pow)


if __name__ == "__main__":
    main()