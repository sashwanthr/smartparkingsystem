import streamlit as st
import sys
import subprocess

# Handle OpenCV import with fallback
try:
    import cv2
except ImportError:
    st.error("OpenCV is not installed. Installing now...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
        import cv2
        st.success("OpenCV installed successfully!")
    except Exception as e:
        st.error(f"Failed to install OpenCV: {e}")
        st.stop()

# Now import other dependencies
try:
    import json
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import time
    import random
    from datetime import datetime, timedelta
    import threading
    import queue
    from pathlib import Path
    import tempfile
    import os
    from typing import Dict, List, Tuple, Optional
    import base64
    import subprocess
    import shutil
    from collections import defaultdict
except ImportError as e:
    st.error(f"Missing dependency: {e}")
    st.stop()

# Handle YOLO import
try:
    from ultralytics import YOLO
except ImportError:
    st.error("YOLO (ultralytics) is not installed. Installing now...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        from ultralytics import YOLO
        st.success("YOLO installed successfully!")
    except Exception as e:
        st.error(f"Failed to install YOLO: {e}")
        st.stop()

# Page configuration
st.set_page_config(
    page_title="Smart Parking Management System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        padding-left: 1rem;
    }
    .status-free {
        color: green;
        font-weight: bold;
    }
    .status-occupied {
        color: red;
        font-weight: bold;
    }
    .status-transition {
        color: orange;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 16px;
    }
    .emergency-stop {
        background-color: #f44336 !important;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'page' not in st.session_state:
    st.session_state.page = 'Configuration'
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'detection_thread' not in st.session_state:
    st.session_state.detection_thread = None

class MultiFloorParkingDetector:
    def __init__(self, model_path='yolov8n.pt', 
                 ground_slots_json='slots1.json', 
                 first_floor_slots_json='slots2.json',
                 ground_video='parking1.mp4',
                 first_floor_video='parking2.mp4',
                 confidence_threshold=0.5):
        """Initialize the multi-floor parking detector with transition states"""
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        
        # Video paths
        self.ground_video = ground_video
        self.first_floor_video = first_floor_video
        
        # Load parking slots for both floors
        print(f"Loading ground floor slots from {ground_slots_json}...")
        self.ground_floor_slots = self.load_parking_slots(ground_slots_json, floor_id="Ground")
        
        print(f"Loading first floor slots from {first_floor_slots_json}...")
        self.first_floor_slots = self.load_parking_slots(first_floor_slots_json, floor_id="1st Floor")
        
        # All slots combined for easy processing
        self.all_slots = self.ground_floor_slots + self.first_floor_slots
        
        # Enhanced parameters with transition states
        self.detection_params = {
            'conf_threshold': confidence_threshold,
            'iou_threshold': 0.45,
            'overlap_threshold': 0.15,
            'min_overlap_for_occupation': 0.25,
            'min_car_area': 2000,
            'stability_frames': 2,
            'transition_time_seconds': 10,
            'fps': 24,
        }
        
        # Transition time in frames
        self.transition_frames = self.detection_params['transition_time_seconds'] * self.detection_params['fps']
        
        # Enhanced tracking with transition states
        self.slot_history = defaultdict(list)
        self.frame_count = 0
        self.is_running = False
        self.processing_queue = queue.Queue(maxsize=2)
        
        # Video captures
        self.cap_ground = None
        self.cap_first = None
        
        # Colors for visualization
        self.colors = {
            'free': (0, 255, 0),           # Green
            'occupied': (0, 0, 255),       # Red
            'transition': (0, 165, 255),   # Orange (BGR format)
            'uncertain': (0, 255, 255),    # Yellow
            'car_box': (255, 255, 0),      # Cyan for cars
            'text_bg': (0, 0, 0),          # Black background
            'text': (255, 255, 255),       # White text
        }
        
        print(f"Multi-floor system initialized:")
        print(f"Ground floor slots: {len(self.ground_floor_slots)}")
        print(f"First floor slots: {len(self.first_floor_slots)}")
        print(f"Transition time: {self.detection_params['transition_time_seconds']} seconds")
    
    def load_parking_slots(self, json_path, floor_id):
        """Load and validate parking slot polygons for a specific floor"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            parking_slots = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                slots_data = data
            elif 'slots' in data:
                slots_data = data['slots']
            elif 'parking_slots' in data:
                slots_data = data['parking_slots']
            else:
                slots_data = [data]
            
            for i, slot in enumerate(slots_data):
                try:
                    # Extract coordinates
                    points = None
                    if isinstance(slot, list):
                        points = slot
                    elif 'points' in slot:
                        points = slot['points']
                    elif 'polygon' in slot:
                        points = slot['polygon']
                    elif 'coordinates' in slot:
                        points = slot['coordinates']
                    elif 'vertices' in slot:
                        points = slot['vertices']
                    
                    if points is None:
                        continue
                    
                    # Convert and validate polygon
                    polygon = np.array(points, dtype=np.int32)
                    if len(polygon.shape) == 1:
                        polygon = polygon.reshape(-1, 2)
                    
                    # Calculate polygon properties
                    area = cv2.contourArea(polygon)
                    
                    # Calculate center and bounding box
                    M = cv2.moments(polygon)
                    center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])) if M['m00'] != 0 else (0, 0)
                    
                    x, y, w, h = cv2.boundingRect(polygon)
                    
                    slot_info = {
                        'id': f"{floor_id}_S{slot.get('id', i + 1) if isinstance(slot, dict) else i + 1}",
                        'floor': floor_id,
                        'polygon': polygon,
                        'area': area,
                        'center': center,
                        'bbox': (x, y, x + w, y + h),
                        'state': 'free',
                        'confidence': 0.0,
                        'stable': True,
                        'last_detection_frame': -1,
                        'associated_car': None,
                        'transition_start_frame': None,
                        'previous_stable_state': 'free',
                        'detection_history': []
                    }
                    parking_slots.append(slot_info)
                    
                except Exception as e:
                    print(f"Error processing slot {i} on {floor_id}: {e}")
                    continue
            
            print(f"Successfully loaded {len(parking_slots)} parking slots for {floor_id}")
            return parking_slots
            
        except Exception as e:
            print(f"Error loading parking slots from {json_path}: {e}")
            return []
    
    def detect_cars(self, frame):
        """Enhanced car detection with optimized performance"""
        results = self.model(
            frame, 
            conf=self.detection_params['conf_threshold'],
            iou=self.detection_params['iou_threshold'],
            verbose=False
        )
        
        car_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Detect vehicles: car, motorcycle, bus, truck, bicycle
                    vehicle_classes = [1, 2, 3, 5, 7]
                    
                    if class_id in vehicle_classes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        width, height = x2 - x1, y2 - y1
                        area = width * height
                        
                        if area >= self.detection_params['min_car_area'] and width > 20 and height > 20:
                            car_detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'class_id': class_id,
                                'area': area,
                                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                                'width': width,
                                'height': height
                            })
        
        return car_detections
    
    def calculate_improved_overlap(self, car_bbox, slot):
        """Improved overlap calculation"""
        x1, y1, x2, y2 = car_bbox
        car_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        slot_polygon = slot['polygon']
        
        # Method 1: Intersection calculation
        car_polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        
        try:
            intersection_result = cv2.intersectConvexConvex(car_polygon, slot_polygon)
            if intersection_result[0] > 0 and intersection_result[1] is not None:
                intersection_area = cv2.contourArea(intersection_result[1])
                car_area = (x2 - x1) * (y2 - y1)
                slot_area = slot['area']
                
                car_overlap = intersection_area / car_area if car_area > 0 else 0
                slot_overlap = intersection_area / slot_area if slot_area > 0 else 0
                intersection_score = max(car_overlap, slot_overlap)
            else:
                intersection_score = 0.0
        except:
            intersection_score = 0.0
        
        # Method 2: Center-based detection
        distance = cv2.pointPolygonTest(slot_polygon, car_center, True)
        
        if distance >= 0:
            max_distance = max(slot['bbox'][2] - slot['bbox'][0], slot['bbox'][3] - slot['bbox'][1]) / 2
            center_score = min(1.0, (distance + 10) / max_distance) if max_distance > 0 else 1.0
        else:
            center_score = max(0.0, 1.0 + distance / 100.0)
        
        # Method 3: Bounding box overlap
        slot_x1, slot_y1, slot_x2, slot_y2 = slot['bbox']
        
        overlap_x1 = max(x1, slot_x1)
        overlap_y1 = max(y1, slot_y1)
        overlap_x2 = min(x2, slot_x2)
        overlap_y2 = min(y2, slot_y2)
        
        if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
            car_area = (x2 - x1) * (y2 - y1)
            bbox_overlap = overlap_area / car_area if car_area > 0 else 0
        else:
            bbox_overlap = 0.0
        
        # Combine scores
        final_score = (
            intersection_score * 0.4 +
            center_score * 0.4 +
            bbox_overlap * 0.2
        )
        
        return final_score
    
    def match_cars_to_slots(self, car_detections, slots):
        """Match cars to slots for a specific floor"""
        slot_detections = []
        used_cars = set()
        
        # Create car-slot pairs with scores
        pairs = []
        for car_idx, car in enumerate(car_detections):
            for slot in slots:
                overlap_score = self.calculate_improved_overlap(car['bbox'], slot)
                if overlap_score >= self.detection_params['overlap_threshold']:
                    pairs.append({
                        'car_idx': car_idx,
                        'slot_id': slot['id'],
                        'score': overlap_score,
                        'car': car,
                        'slot': slot
                    })
        
        pairs.sort(key=lambda x: x['score'], reverse=True)
        used_slots = set()
        
        for pair in pairs:
            if pair['car_idx'] not in used_cars and pair['slot_id'] not in used_slots:
                slot_detections.append({
                    'slot_id': pair['slot_id'],
                    'car': pair['car'],
                    'overlap_ratio': pair['score'],
                    'is_occupied': pair['score'] >= self.detection_params['min_overlap_for_occupation']
                })
                used_cars.add(pair['car_idx'])
                used_slots.add(pair['slot_id'])
        
        return slot_detections
    
    def update_slot_status_with_transitions(self, slot_detections, slots):
        """Update slot status with transition states"""
        current_frame = self.frame_count
        slot_detection_map = {d['slot_id']: d for d in slot_detections}
        
        for slot in slots:
            slot_id = slot['id']
            
            # Current detection status
            current_detection = slot_detection_map.get(slot_id)
            is_currently_occupied = bool(current_detection and current_detection['is_occupied'])
            
            # Add to detection history
            slot['detection_history'].append(is_currently_occupied)
            
            # Keep only recent history (last 5 frames for stability check)
            if len(slot['detection_history']) > 5:
                slot['detection_history'] = slot['detection_history'][-5:]
            
            # Update associated car
            slot['associated_car'] = current_detection['car'] if current_detection and is_currently_occupied else None
            slot['confidence'] = current_detection['overlap_ratio'] if current_detection else 0.0
            
            current_state = slot['state']
            
            # Handle state transitions
            if current_state == 'free':
                if is_currently_occupied:
                    slot['state'] = 'transition_to_occupied'
                    slot['transition_start_frame'] = current_frame
                    slot['stable'] = False
                    
            elif current_state == 'occupied':
                if not is_currently_occupied:
                    slot['state'] = 'transition_to_free'
                    slot['transition_start_frame'] = current_frame
                    slot['stable'] = False
                    
            elif current_state == 'transition_to_occupied':
                frames_in_transition = current_frame - slot.get('transition_start_frame', current_frame)
                
                if frames_in_transition >= self.transition_frames:
                    recent_occupied = sum(slot['detection_history'][-3:]) if len(slot['detection_history']) >= 3 else 0
                    
                    if recent_occupied >= 2:
                        slot['state'] = 'occupied'
                        slot['previous_stable_state'] = 'occupied'
                        slot['stable'] = True
                    else:
                        slot['state'] = 'free'
                        slot['previous_stable_state'] = 'free'
                        slot['stable'] = True
                else:
                    if not is_currently_occupied and frames_in_transition > self.detection_params['fps'] * 2:
                        slot['state'] = 'free'
                        slot['stable'] = True
                        
            elif current_state == 'transition_to_free':
                frames_in_transition = current_frame - slot.get('transition_start_frame', current_frame)
                
                if frames_in_transition >= self.transition_frames:
                    recent_occupied = sum(slot['detection_history'][-3:]) if len(slot['detection_history']) >= 3 else 0
                    
                    if recent_occupied <= 1:
                        slot['state'] = 'free'
                        slot['previous_stable_state'] = 'free'
                        slot['stable'] = True
                    else:
                        slot['state'] = 'occupied'
                        slot['previous_stable_state'] = 'occupied'
                        slot['stable'] = True
                else:
                    if is_currently_occupied and frames_in_transition > self.detection_params['fps'] * 2:
                        slot['state'] = 'occupied'
                        slot['stable'] = True
    
    def draw_floor_results(self, frame, car_detections, slots, floor_name):
        """Draw results for a specific floor with transition states"""
        result_frame = frame.copy()
        
        # Draw parking slots with transition colors
        for slot in slots:
            # Choose color and status based on state
            if slot['state'] == 'free':
                color = self.colors['free']
                thickness = 3
                status_text = "FREE"
            elif slot['state'] == 'occupied':
                color = self.colors['occupied']
                thickness = 4
                status_text = "OCCUPIED"
            elif slot['state'] in ['transition_to_occupied', 'transition_to_free']:
                color = self.colors['transition']
                thickness = 3
                frames_passed = self.frame_count - slot.get('transition_start_frame', self.frame_count)
                frames_remaining = max(0, self.transition_frames - frames_passed)
                seconds_remaining = max(0, frames_remaining // self.detection_params['fps'])
                
                if slot['state'] == 'transition_to_occupied':
                    status_text = f"PARKING ({seconds_remaining}s)"
                else:
                    status_text = f"LEAVING ({seconds_remaining}s)"
            else:
                color = self.colors['uncertain']
                thickness = 2
                status_text = "CHECKING"
            
            # Draw polygon outline
            cv2.polylines(result_frame, [slot['polygon']], True, color, thickness)
            
            # Semi-transparent fill
            overlay = result_frame.copy()
            cv2.fillPoly(overlay, [slot['polygon']], color)
            cv2.addWeighted(result_frame, 0.85, overlay, 0.15, 0, result_frame)
            
            # Status text
            center_x, center_y = slot['center']
            slot_id = slot['id'].split('_')[-1]
            
            lines = [f"{slot_id}", f"{status_text}", f"({slot['confidence']:.2f})"]
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            
            for i, line in enumerate(lines):
                text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                text_x = center_x - text_size[0] // 2
                text_y = center_y - 20 + i * 15
                
                # Text background
                cv2.rectangle(result_frame,
                            (text_x - 2, text_y - 12),
                            (text_x + text_size[0] + 2, text_y + 2),
                            self.colors['text_bg'], -1)
                
                # Text
                cv2.putText(result_frame, line, (text_x, text_y),
                          font, font_scale, self.colors['text'], thickness)
        
        # Draw car detections
        class_names = {1: 'Bike', 2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
        
        for car in car_detections:
            x1, y1, x2, y2 = car['bbox']
            confidence = car['confidence']
            
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), self.colors['car_box'], 2)
            
            vehicle_name = class_names.get(car['class_id'], 'Vehicle')
            label = f"{vehicle_name} {confidence:.2f}"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(result_frame,
                        (x1, y1 - label_size[1] - 5),
                        (x1 + label_size[0], y1),
                        self.colors['car_box'], -1)
            
            cv2.putText(result_frame, label, (x1, y1 - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Floor statistics
        total_slots = len(slots)
        free_slots = sum(1 for slot in slots if slot['state'] == 'free')
        occupied_slots = sum(1 for slot in slots if slot['state'] == 'occupied')
        transition_slots = sum(1 for slot in slots if slot['state'] in ['transition_to_occupied', 'transition_to_free'])
        
        # Statistics display
        stats_height = 70
        cv2.rectangle(result_frame, (10, 10), (550, stats_height), (0, 0, 0), -1)
        
        stats_lines = [
            f"{floor_name} - Frame: {self.frame_count}",
            f"Free: {free_slots} | Occupied: {occupied_slots} | Changing: {transition_slots}",
            f"Cars: {len(car_detections)} | Total Slots: {total_slots}"
        ]
        
        for i, line in enumerate(stats_lines):
            cv2.putText(result_frame, line,
                      (15, 25 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                      self.colors['text'], 1)
        
        return result_frame
    
    def process_frame_dual(self, ground_frame, first_floor_frame):
        """Process frames from both floors"""
        self.frame_count += 1
        
        # Detect cars on both floors
        ground_car_detections = self.detect_cars(ground_frame)
        first_floor_car_detections = self.detect_cars(first_floor_frame)
        
        # Match cars to slots for each floor
        ground_slot_detections = self.match_cars_to_slots(ground_car_detections, self.ground_floor_slots)
        first_floor_slot_detections = self.match_cars_to_slots(first_floor_car_detections, self.first_floor_slots)
        
        # Update slot statuses with transition logic
        self.update_slot_status_with_transitions(ground_slot_detections, self.ground_floor_slots)
        self.update_slot_status_with_transitions(first_floor_slot_detections, self.first_floor_slots)
        
        # Draw results
        ground_result = self.draw_floor_results(ground_frame, ground_car_detections, self.ground_floor_slots, "Ground Floor")
        first_floor_result = self.draw_floor_results(first_floor_frame, first_floor_car_detections, self.first_floor_slots, "1st Floor")
        
        return ground_result, first_floor_result
    
    def download_video_if_needed(self, video_path, file_id):
        """Download large video files from Google Drive if they don't exist locally"""
        if os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            if file_size > 1024:  # File exists and is not empty
                return True
            else:
                # Remove corrupted file
                os.remove(video_path)
        
        try:
            import urllib.request
            import urllib.parse
            import re
            
            print(f"Downloading {video_path} from Google Drive...")
            
            # Handle large files that require confirmation
            session_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            # First request to get the confirmation token for large files
            response = urllib.request.urlopen(session_url)
            content = response.read().decode('utf-8')
            
            # Look for the confirmation token
            confirm_token = None
            if 'confirm=' in content:
                token_match = re.search(r'confirm=([a-zA-Z0-9_-]+)', content)
                if token_match:
                    confirm_token = token_match.group(1)
            
            # Download URL with confirmation token if needed
            if confirm_token:
                download_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                print("Large file detected, using confirmation token...")
            else:
                download_url = session_url
            
            urllib.request.urlretrieve(download_url, video_path)
            
            # Verify download
            if os.path.exists(video_path) and os.path.getsize(video_path) > 1024:
                file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                print(f"Downloaded {video_path} successfully ({file_size_mb:.1f}MB)")
                return True
            else:
                print(f"Download failed: {video_path} is too small or corrupted")
                return False
                
        except Exception as e:
            print(f"Error downloading {video_path}: {e}")
            return False
    
    def start_detection(self):
        """Start video detection with automatic video download and better error handling"""
        if self.is_running:
            return True
            
        # Google Drive file IDs for your videos
        video_files = {
            'parking1.mp4': '1yVBgp07Z8FfLzw5dd0SxW61zd1l55kjE',  # Ground floor
            'parking2.mp4': '14njmPC4b81mOV02orKslziMRr2WL-qAM'   # First floor
        }
        
        # Download and verify videos if they don't exist or are corrupted
        for video_file, file_id in video_files.items():
            if not os.path.exists(video_file) or os.path.getsize(video_file) <= 1024:
                print(f"Downloading {video_file}...")
                if os.path.exists(video_file):
                    os.remove(video_file)  # Remove corrupted file
                if not self.download_video_if_needed(video_file, file_id):
                    print(f"Failed to download {video_file}")
                    return False
        
        # Verify files exist and have reasonable size
        if not os.path.exists(self.ground_video) or os.path.getsize(self.ground_video) <= 1024:
            print(f"Ground floor video {self.ground_video} is missing or corrupted")
            return False
            
        if not os.path.exists(self.first_floor_video) or os.path.getsize(self.first_floor_video) <= 1024:
            print(f"First floor video {self.first_floor_video} is missing or corrupted")
            return False
        
        try:
            self.cap_ground = cv2.VideoCapture(self.ground_video)
            self.cap_first = cv2.VideoCapture(self.first_floor_video)
            
            if not self.cap_ground.isOpened():
                print(f"Error: Cannot open ground floor video {self.ground_video}")
                return False
                
            if not self.cap_first.isOpened():
                print(f"Error: Cannot open first floor video {self.first_floor_video}")
                self.cap_ground.release()
                return False
            
            # Test read a frame to ensure videos are readable
            ret1, _ = self.cap_ground.read()
            ret2, _ = self.cap_first.read()
            
            if not ret1 or not ret2:
                print("Error: Cannot read frames from video files")
                self.cap_ground.release()
                self.cap_first.release()
                return False
            
            # Reset to beginning
            self.cap_ground.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.cap_first.set(cv2.CAP_PROP_POS_FRAMES, 0)
