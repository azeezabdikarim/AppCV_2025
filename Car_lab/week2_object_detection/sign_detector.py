#!/usr/bin/env python3

import cv2
import numpy as np
import os
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.utils.console_logger import console_logger


class SignDetector:
    """
    WEEK 2: PERCEPTION FOR AUTONOMOUS SYSTEMS
    ==========================================
    
    THE BIG PICTURE:
    In autonomous vehicles, we utilize multiple AI models to compose complex decision making. 
    Think about our porgram as a conductor in an orchestra, telling different instuments (tools) to come in and provide
    their sound (functionality) at the necessary moments of the song (decison process).
    
    PERCEPTION PIPELINE:
    Camera → Object Detection → Depth Estimation → Decision Logic → Robot Action
    
    TODAY'S JOURNEY:
    1. 'detect_signs()' - Use YOLO model to find objects in camera images
    2. 'detect_signs()' - Understand model outputs and tune confidence thresholds  
    3. 'should_stop()' - Make stopping decisions using bounding box size
    4. '_advanced_depth_stopping()' - Upgrade stopping decision to use the MiDaS monocular depth estimateion model
    
    LEARNING GOALS:
    - AI models are tools: understand inputs/outputs and how to apply these tools to navigate complex problems
    - Multiple models can work together
    - Threshold tuning is critical for real-world performance
    - Understand the perfomrance of simple implmentations before adding complexity

    NOTE: Use console_logger.stop() instead of print() for important messages - they'll appear in the web console!
    """
    
    def __init__(self):
        """Initialize ONNX model and detection parameters"""
        
        # Model and detection setup
        self.model_path = os.path.join(os.path.dirname(__file__), 'models', 'best.onnx')
        self.class_names = ['Stop_Sign', 'TU_Logo', 'Stahp', 'Falling_Cows']
        
        # STUDENT TUNABLE PARAMETERS - Experiment with these values
        self.confidence_threshold = 0.05      # Minimum confidence to trust detections
        self.simple_area_threshold = 5000    # Pixel area threshold for stopping
        self.depth_distance_threshold = 1.5  # Distance threshold in meters (advanced)
        
        # Model input specifications (determined by training)
        self.input_size = 640  # Model expects 640x640 input
        
        # Initialize caching for debug visualization
        self._cached_depth_map = None
        self._last_depth_inference_time = 0
        
        # Load the ONNX model
        self._load_model()
        
        # Try to load depth estimation model (optional for advanced section)
        self._load_depth_model()
        
        print("SignDetector initialized - Students: Implement detect_signs() and should_stop()!")

    def detect_signs(self, camera_frame):
        """
        STEP 1: OBJECT DETECTION PIPELINE
        =================================
        
        CONCEPT: Transform camera image → list of detected objects
        
        THE PROCESS:
        1. Preprocess: Camera format → Model format (we provide this)
        2. Inference: Run neural network (YOU implement this)  
        3. Investigate: Explore what the model outputs (YOU discover this)
        4. Parse: Extract useful information (YOU implement this)
        5. Filter: Keep only confident detections (YOU decide thresholds)
        """
        
        if self.session is None:
            return []  # No model loaded
        
        try:
            # STEP 1: Preprocessing (provided for you)
            # Converts 320x240 BGR camera image → to the tensor format needed for our Yolov8 nanon model
            input_tensor = self._preprocess_frame(camera_frame)
            if input_tensor is None:
                return []
            
            print(f"Preprocessed frame shape: {input_tensor.shape}")
            
            # STEP 2: INFERENCE - YOU IMPLEMENT THIS LINE
            # TASK: Run the neural network on your preprocessed image
            # TODO: Call self.session.run() with the input tensor
            # HINT: outputs = self.session.run(None, {self.input_name: input_tensor})
            
            outputs = None  # TODO: Replace this line with actual inference call
            
            if outputs is None:
                return [] 
            
            # STEP 3: INVESTIGATE THE OUTPUTS
            # DISCOVERY TASK: What did the model give us back?
            # print(f"Model output shape: {outputs[0].shape}")
            # print(f"Output data type: {outputs[0].dtype}")
            
            # QUESTION: What do you think these dimensions mean?
            # Our model vocabulary: ['Stop_Sign', 'TU_Logo', 'Stahp', 'Falling_Cows'] (4 classes)
            # Expected shape: [1, 8, 8400] = [batch, coordinates+classes, detections]
            
            # STEP 4: PARSE THE OUTPUTS - YOU IMPLEMENT THIS
            # The output format is: [center_x, center_y, width, height, class0_conf, class1_conf, class2_conf, class3_conf]
            
            detections = [] # This is where we will store properly formated boundary boxes we detect
            
            # TODO: Reshape the outputs to work with them. Currenlty our outputs is shapped like [1,8, n], but we want ot pupulate our 'detections' list iwht list structed as [n, 8] 
            # HINT: One way of going from [1, 8, n] to [n, 8] ......
            # pred = outputs[0]  # Remove batch dimension
            # pred = np.transpose(pred, (1, 0))  # Transpose to [8400, 8]
            
            # TODO: Loop through each detection
            # for detection in pred:
            #     # TODO: Extract coordinates (first 4 values from detection)
            #     # center_x, center_y, width, height = 
            #     
            #     # TODO: Extract confidence scores (last 4 values from detection)  
            #     # class_confidences =
            #     
            #     # TODO: Find the class with highest confidence
            #     # max_confidence = np.max(class_confidences)
            #     # predicted_class_id = np.argmax(class_confidences)
            #     
            #     # TODO: Filter by confidence threshold
            #     # if max_confidence > self.confidence_threshold:
            #         # TODO: Convert center format to corner format and scale coordinates
            #         # converted_bbox = self._convert_coordinates(center_x, center_y, width, height, camera_frame.shape)
            #         # 
            #         # TODO: Add to detections list in required format
            #         # detections.append({
            #         #     'bbox': converted_bbox,
            #         #     'confidence': float(max_confidence),
            #         #     'class_name': self.class_names[predicted_class_id]
            #         # })
            
            # REQUIRED OUTPUT FORMAT:
            # [{'bbox': [x, y, w, h], 'confidence': 0.95, 'class_name': 'Stop_Sign'}, ...]
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def should_stop(self, detected_signs, camera_frame):
        """
        STEP 2: DECISION LOGIC PIPELINE  
        ===============================
        
        CONCEPT: Convert perception data to robot actions
        
        THE CORE INSIGHT:
        How do we estimate distance from a single 2D camera image?
        
        APPROACH 1 - SIZE-BASED REASONING (Simple & Effective):
        - Objects appear larger when closer to camera
        - Measure bounding box area in pixels  
        - Large area = close object = STOP
        - Small area = far object = CONTINUE
        
        APPROACH 2 - DEPTH ESTIMATION (Advanced):
        - Use an AI model to estimate depth at every pixel
        - Sample depths within the bounding box
        - If depth estimates show we're close to the object...... STOP!
        
        IMPORTANT: We must check ALL detected objects
        - Multiple signs might be detected in one frame
        - If any object is close enough, stop the robot
        - Iterate through the entire detected_signs list
        """
        
        if not detected_signs:
            return False  # No objects detected, safe to continue
        
        # APPROACH 1: SIZE-BASED STOPPING (Implement this first)
        # CONCEPT: Bigger bounding box = closer object
        
        # TASK: Check ALL detected objects to see if ANY are close enough to stop
        # TODO: Loop through detected_signs and check each detection
        # for detection in detected_signs:
        #     bbox = detection['bbox']  # [x, y, width, height]
        #     
        #     # TODO: Calculate area of this detection using the width and the heigh
        #     # area = 
        #     
        #     # TODO: Compare to threshold
        #     # if area > self.simple_area_threshold:
        #         # print(f"STOPPING: {detection['class_name']} area {area} > threshold {self.simple_area_threshold}")
        #         # return True
        
        # PLACEHOLDER: Remove this when you implement the loop above
        largest_detection = None  # Replace this
        largest_area = 0
        
        # TODO: Calculate area and compare to threshold
        # EXPERIMENT: Try different thresholds (start with 5000 pixels)
        # - Too low = stops too far away
        # - Too high = doesn't stop until very close
        
        area_threshold = self.simple_area_threshold  # Tune this parameter
        
        if largest_area > area_threshold:
            print(f"🛑 STOPPING: Object area {largest_area} > threshold {area_threshold}")
            return True
        
        # APPROACH 2: DEPTH-BASED STOPPING (Advanced - implement after area works)
        # CONCEPT: Use actual relative distance measurements that come from our monocular depth estimation package
        
        # UNCOMMENT THIS SECTION WHEN USING THE ADVANCED APPROACH, also comment out the return statmeent above that's based on the area threshold
        # return self._advanced_depth_stopping(detected_signs, camera_frame)
        
        return False
    
    def _advanced_depth_stopping(self, detected_signs, camera_frame):
        """
        ADVANCED: DEPTH-BASED DISTANCE ESTIMATION
        =========================================
        
        CONCEPT: Get actual distance measurements using depth estimation
        
        THE PROCESS:
        1. Run depth estimation model on camera frame
        2. For each detected object, sample depth within its bounding box
        3. Calculate average/median depth → real distance
        4. Stop if distance < threshold
        
        DEBUGGING INTEGRATION:
        - This function caches depth results for visualization
        - When you implement this, depth maps will appear in debug panel
        """
        
        if self.depth_estimator is None:
            print("⚠️  Depth estimator not available, falling back to area-based method")
            return False
        
        try:
            # STEP 1: RUN DEPTH MODEL - YOU IMPLEMENT THIS LINE
            # TODO: Get depth map from MiDaS model
            # HINT: depth_map = self.depth_estimator.predict(camera_frame)
            
            depth_map = None  # TODO: Replace this line with actual depth inference
            
            if depth_map is None:
                return False  # Fallback to area-based method
            
            # VERY IMPORTANT! VERY IMPORTANT FOR DEBUGGING!
            # VERY IMPORTANT! VERY IMPORTANT FOR DEBUGGING!
            # VERY IMPORTANT! VERY IMPORTANT FOR DEBUGGING!
            # VERY IMPORTANT! VERY IMPORTANT FOR DEBUGGING!
            # DEBUGGING: Cache depth results for visualization 
            # UNCOMMENT the line below to see depth maps in debug panel
            # self._cached_depth_map = depth_map
            
            # STEP 2: SAMPLE DEPTH VALUES FOR EACH DETECTION
            # DISCOVERY QUESTION: Where in the depth map should you look?
            # ANSWER: Look at pixels inside the detected bounding boxes!
            
            # IMPORTANT: Check ALL detected objects, not just the largest one
            for detection in detected_signs:
                bbox = detection['bbox']  # [x, y, w, h]
                x, y, w, h = bbox
                
                # TODO: Extract depth values within bounding box
                # HINT: depth_region = depth_map[y:y+h, x:x+w]
                
                # DISCOVERY QUESTION: How do you convert depth map values to real distances?
                # EXPERIMENT: Print depth values and see what range they have
                # HINT: Smaller depth values often mean closer objects
                
                # TODO: Calculate representative depth (mean, median, minimum?)
                # representative_depth = np.mean(depth_region)  # or np.median, np.min
                
                # TODO: Convert to real-world distance if needed
                # estimated_distance = self._depth_to_distance(representative_depth)
                
                # TODO: Compare to distance threshold
                # if estimated_distance < self.depth_distance_threshold:
                #     console_logger.stop(f"🛑 STOPPING: {detection['class_name']} at {estimated_distance:.1f}m < {self.depth_distance_threshold}m")
                #     return True
            
            return False  # No objects close enough to stop
            
        except Exception as e:
            print(f"Depth analysis error: {e}")
            return False
    
    def _convert_coordinates(self, center_x, center_y, width, height, original_shape):
        """
        Helper function: Convert model coordinates to image coordinates (provided)
        
        TECHNICAL DETAILS:
        - Convert center format [cx, cy, w, h] to corner format [x, y, w, h]
        - Scale from model size (640x640) back to original camera size (320x240)
        """
        orig_height, orig_width = original_shape[:2]
        
        # Convert center format to corner format
        x = center_x - width / 2
        y = center_y - height / 2
        
        # Scale coordinates from model size to original image size
        scale_x = orig_width / self.input_size
        scale_y = orig_height / self.input_size
        
        x = int(x * scale_x)
        y = int(y * scale_y)
        width = int(width * scale_x)
        height = int(height * scale_y)
        
        # Ensure coordinates are within image bounds
        x = max(0, min(x, orig_width - 1))
        y = max(0, min(y, orig_height - 1))
        width = min(width, orig_width - x)
        height = min(height, orig_height - y)
        
        return [x, y, width, height]

    def _depth_to_distance(self, depth_value):
        """
        Helper function: Convert depth map value to real-world distance (provided)
        
        TECHNICAL NOTE:
        This conversion depends on the specific depth model and calibration.
        MiDaS outputs relative depth, so this is a simplified conversion.
        """
        # Simple conversion - students can experiment with this
        # MiDaS typically outputs inverse depth, so smaller values = closer objects
        if depth_value <= 0:
            return float('inf')  # Invalid depth
        
        # Empirical conversion (you may need to calibrate this for your setup)
        estimated_distance = 1.0 / (depth_value + 0.1)  # Avoid division by zero
        return estimated_distance
    
    def _load_model(self):
        """Load the ONNX model (provided helper function)"""
        try:
            import onnxruntime as ort
            
            if os.path.exists(self.model_path):
                # Optimize for CPU performance
                providers = ['CPUExecutionProvider']
                sess_options = ort.SessionOptions()
                sess_options.inter_op_num_threads = 2
                sess_options.intra_op_num_threads = 2
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                self.session = ort.InferenceSession(
                    self.model_path, 
                    sess_options=sess_options,
                    providers=providers
                )
                
                self.input_name = self.session.get_inputs()[0].name
                self.input_shape = self.session.get_inputs()[0].shape
                
                print(f"✅ ONNX model loaded: {self.model_path}")
                print(f"   Model input shape: {self.input_shape}")
                print(f"   Model classes: {self.class_names}")
            else:
                print(f"❌ Model file not found: {self.model_path}")
                self.session = None
                
        except ImportError:
            print("❌ ONNX Runtime not available. Please install: pip install onnxruntime")
            self.session = None
        except Exception as e:
            print(f"❌ Error loading ONNX model: {e}")
            self.session = None
    
    def _load_depth_model(self):
        """Try to load depth estimation model (optional for advanced section)"""
        try:
            # Students will install this when they reach the advanced section
            from midas_depth import DepthEstimator
            self.depth_estimator = DepthEstimator()
            print("✅ Depth estimation model loaded (advanced features available)")
        except ImportError:
            self.depth_estimator = None
            print("⚪ Depth estimation not available (install midas_depth for advanced features)")
        except Exception as e:
            self.depth_estimator = None
            print(f"⚠️  Depth model loading failed: {e}")
    
    def _preprocess_frame(self, camera_frame):
        """
        Preprocess camera frame for ONNX model input (provided helper function)
        
        TECHNICAL DETAILS (students don't need to implement this):
        - Resize 320x240 camera frame to 640x640 model input
        - Convert BGR to RGB color space
        - Normalize pixel values from 0-255 to 0.0-1.0
        - Rearrange dimensions for neural network format
        """
        if camera_frame is None:
            return None
            
        # Resize to model input size
        resized = cv2.resize(camera_frame, (self.input_size, self.input_size))
        
        # Convert to float and normalize
        input_tensor = resized.astype(np.float32)
        input_tensor /= 255.0
        
        # Rearrange dimensions: HWC → CHW → BCHW
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC to CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)   # Add batch dimension
        
        return input_tensor
    
    def get_cached_depth_map(self):
        """Return cached depth map for debug visualization (provided)"""
        return self._cached_depth_map