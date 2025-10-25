import streamlit as st
import cv2
import numpy as np
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
import tempfile
from typing import Dict, List, Tuple, Optional
from collections import deque

# Set page config first
st.set_page_config(
    page_title="Multi-Floor Parking Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .status-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1e3c72;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .sidebar-status {
        background: #2d3748;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        color: #ffffff;
    }
    .video-container {
        border: 2px solid #1e3c72;
        border-radius: 8px;
        padding: 0.5rem;
        background: white;
    }
    .slot-free { color: #28a745; font-weight: bold; }
    .slot-occupied { color: #dc3545; font-weight: bold; }
    .slot-transition { color: #fd7e14; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'detection_active': False,
        'detector': None,
        'video_captures': {},
        'frame_counts': {},
        'last_update': datetime.now(),
        'config': {
            'confidence_threshold': 0.25,
            'transition_time': 10,
            'ground_floor_video': None,
            'first_floor_video': None,
            'ground_floor_slots': None,
            'first_floor_slots': None
        },
        'temp_dir': tempfile.mkdtemp(),
        # Real-time data tracking
        'historical_data': deque(maxlen=1000),  # Keep last 1000 records
        'slot_history': {},  # Track individual slot changes
        'occupancy_timeline': deque(maxlen=500),  # Real-time occupancy data
        'detection_log': deque(maxlen=100),  # Recent detection events
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Check YOLO availability
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class ParkingSlot:
    """Enhanced parking slot with advanced state management"""
    def __init__(self, slot_id: str, polygon: List[List[int]]):
        self.id = slot_id
        self.polygon = np.array(polygon, dtype=np.int32)
        self.status = "Free"
        self.confidence = 0.0
        self.transition_start_frame = None
        self.transition_duration_frames = 0
        self.status_history = []
        
        # Enhanced properties
        self.area = cv2.contourArea(self.polygon)
        M = cv2.moments(self.polygon)
        self.center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])) if M['m00'] != 0 else (0, 0)
        x, y, w, h = cv2.boundingRect(self.polygon)
        self.bbox = (x, y, x + w, y + h)
        
        self.stable = True
        self.previous_stable_state = 'Free'
        self.detection_history = []
        self.associated_car = None
        
        # Real-time tracking
        self.last_status_change = datetime.now()
        self.occupation_duration = 0
        self.total_occupations = 0
        
    def calculate_overlap(self, car_bbox):
        """Enhanced overlap calculation with multiple methods"""
        x1, y1, x2, y2 = car_bbox
        car_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # Method 1: Intersection calculation
        car_polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        
        try:
            intersection_result = cv2.intersectConvexConvex(car_polygon, self.polygon)
            if intersection_result[0] > 0 and intersection_result[1] is not None:
                intersection_area = cv2.contourArea(intersection_result[1])
                car_area = (x2 - x1) * (y2 - y1)
                
                car_overlap = intersection_area / car_area if car_area > 0 else 0
                slot_overlap = intersection_area / self.area if self.area > 0 else 0
                intersection_score = max(car_overlap, slot_overlap)
            else:
                intersection_score = 0.0
        except:
            intersection_score = 0.0
        
        # Method 2: Center-based detection
        distance = cv2.pointPolygonTest(self.polygon, car_center, True)
        
        if distance >= 0:
            max_distance = max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]) / 2
            center_score = min(1.0, (distance + 10) / max_distance) if max_distance > 0 else 1.0
        else:
            center_score = max(0.0, 1.0 + distance / 100.0)
        
        # Method 3: Bounding box overlap
        slot_x1, slot_y1, slot_x2, slot_y2 = self.bbox
        
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
        
        # Combine scores with weights
        final_score = (
            intersection_score * 0.4 +
            center_score * 0.4 +
            bbox_overlap * 0.2
        )
        
        return final_score
    
    def get_color(self):
        """Get color based on status"""
        colors = {
            "Free": (0, 255, 0),
            "Occupied": (0, 0, 255),
            "Parking": (0, 165, 255),
            "Leaving": (255, 165, 0)
        }
        return colors.get(self.status, (128, 128, 128))
    
    def update_status(self, new_status):
        """Update status and track changes"""
        if self.status != new_status:
            self.last_status_change = datetime.now()
            if new_status == "Occupied":
                self.total_occupations += 1
            self.status = new_status

class MockYOLODetector:
    """Mock YOLO detector for demonstration"""
    
    def __init__(self, confidence_threshold=0.25):
        self.confidence_threshold = confidence_threshold
        self.vehicle_classes = [1, 2, 3, 5, 7]
        
    def detect(self, frame):
        """Generate random vehicle detections"""
        height, width = frame.shape[:2]
        detections = []
        
        num_detections = np.random.randint(3, 8)
        
        for _ in range(num_detections):
            x1 = np.random.randint(0, width - 150)
            y1 = np.random.randint(0, height - 100)
            x2 = x1 + np.random.randint(80, 150)
            y2 = y1 + np.random.randint(50, 100)
            
            detection = {
                'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                'bbox': (x1, y1, x2, y2),
                'confidence': np.random.uniform(0.4, 0.95),
                'class_id': np.random.choice(self.vehicle_classes),
                'area': (x2 - x1) * (y2 - y1)
            }
            
            if detection['confidence'] >= self.confidence_threshold:
                detections.append(detection)
        
        return detections

class EnhancedParkingDetectionSystem:
    """Advanced parking detection system with real-time data tracking"""
    
    def __init__(self):
        self.model = None
        self.mock_detector = None
        self.floors = {}
        self.video_captures = {}
        self.frame_counts = {}
        self.use_mock = True
        
        # Enhanced detection parameters
        self.detection_params = {
            'conf_threshold': 0.25,
            'iou_threshold': 0.45,
            'overlap_threshold': 0.15,
            'min_overlap_for_occupation': 0.25,
            'min_car_area': 2000,
            'transition_time_seconds': 10,
            'fps': 30,
        }
        
        self.transition_frames = self.detection_params['transition_time_seconds'] * self.detection_params['fps']
        self.current_frame = 0
        
        # Real-time data tracking
        self.start_time = datetime.now()
        self.last_log_time = datetime.now()
        
    def load_model(self, confidence_threshold: float = 0.25):
        """Load YOLO model or fallback to mock detector"""
        self.detection_params['conf_threshold'] = confidence_threshold
        
        try:
            if YOLO_AVAILABLE:
                self.model = YOLO('yolov8n.pt')
                self.use_mock = False
                return True
        except Exception as e:
            st.warning(f"YOLO model not available: {e}. Using simulation mode.")
        
        self.mock_detector = MockYOLODetector(confidence_threshold)
        self.use_mock = True
        return True
    
    def load_slots_from_json(self, json_file, floor_name: str) -> int:
        """Load slots from uploaded JSON file - supports multiple formats"""
        try:
            json_data = json.load(json_file)
            slots = {}
            
            # Handle different JSON formats
            if isinstance(json_data, list):
                if len(json_data) > 0 and isinstance(json_data[0], list):
                    prefix = "F" if "First" in floor_name else "G"
                    for idx, polygon in enumerate(json_data):
                        slot_id = f"{prefix}{idx + 1}"
                        slots[slot_id] = ParkingSlot(slot_id, polygon)
                else:
                    for slot_data in json_data:
                        slot_id = slot_data.get('id', slot_data.get('slot_id'))
                        polygon = slot_data.get('polygon', slot_data.get('coordinates', []))
                        if slot_id and polygon:
                            slots[slot_id] = ParkingSlot(slot_id, polygon)
            
            elif isinstance(json_data, dict):
                if "slots" in json_data:
                    slots_list = json_data["slots"]
                    for slot_data in slots_list:
                        slot_id = slot_data.get('id', slot_data.get('slot_id'))
                        polygon = slot_data.get('polygon', slot_data.get('coordinates', []))
                        if slot_id and polygon:
                            slots[slot_id] = ParkingSlot(slot_id, polygon)
                else:
                    for slot_id, slot_info in json_data.items():
                        polygon = slot_info.get("coordinates", slot_info.get("polygon", []))
                        if polygon:
                            slots[slot_id] = ParkingSlot(slot_id, polygon)
            
            self.floors[floor_name] = slots
            
            # Initialize slot history tracking
            for slot_id in slots.keys():
                st.session_state.slot_history[f"{floor_name}_{slot_id}"] = []
            
            # Update transition duration for all slots
            for slot in slots.values():
                slot.transition_duration_frames = self.transition_frames
            
            return len(slots)
            
        except Exception as e:
            st.error(f"Error loading slots from JSON: {e}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")
            return 0
    
    def load_video(self, video_file, floor_name: str) -> bool:
        """Load video file and update FPS settings"""
        try:
            temp_path = os.path.join(st.session_state.temp_dir, f"{floor_name}.mp4")
            with open(temp_path, 'wb') as f:
                f.write(video_file.getbuffer())
            
            cap = cv2.VideoCapture(temp_path)
            if cap.isOpened():
                self.video_captures[floor_name] = cap
                self.frame_counts[floor_name] = 0
                
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                if fps > 0:
                    self.detection_params['fps'] = fps
                    self.transition_frames = self.detection_params['transition_time_seconds'] * fps
                    
                    for floor_slots in self.floors.values():
                        for slot in floor_slots.values():
                            slot.transition_duration_frames = self.transition_frames
                
                return True
            return False
        except Exception as e:
            st.error(f"Error loading video for {floor_name}: {e}")
            return False
    
    def get_video_frame(self, floor_name: str) -> Optional[np.ndarray]:
        """Get next frame from video with looping"""
        if floor_name not in self.video_captures:
            return None
        
        cap = self.video_captures[floor_name]
        ret, frame = cap.read()
        
        if ret:
            self.frame_counts[floor_name] += 1
            return frame
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_counts[floor_name] = 0
            ret, frame = cap.read()
            return frame if ret else None
    
    def detect_vehicles_in_frame(self, frame: np.ndarray) -> List[dict]:
        """Enhanced vehicle detection with YOLO or mock"""
        if self.use_mock or self.model is None:
            return self.mock_detector.detect(frame)
        
        results = self.model(
            frame,
            conf=self.detection_params['conf_threshold'],
            iou=self.detection_params['iou_threshold'],
            verbose=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    vehicle_classes = [1, 2, 3, 5, 7]
                    
                    if class_id in vehicle_classes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        width, height = x2 - x1, y2 - y1
                        area = width * height
                        
                        if area >= self.detection_params['min_car_area'] and width > 20 and height > 20:
                            detections.append({
                                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'class_id': class_id,
                                'area': area
                            })
        
        return detections
    
    def match_cars_to_slots(self, car_detections: List[dict], floor_name: str):
        """Match cars to slots using improved overlap calculation"""
        if floor_name not in self.floors:
            return []
        
        slots = self.floors[floor_name]
        slot_detections = []
        used_cars = set()
        
        pairs = []
        for car_idx, car in enumerate(car_detections):
            for slot_id, slot in slots.items():
                overlap_score = slot.calculate_overlap(car['bbox'])
                if overlap_score >= self.detection_params['overlap_threshold']:
                    pairs.append({
                        'car_idx': car_idx,
                        'slot_id': slot_id,
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
    
    def update_floor_slots(self, floor_name: str, frame: np.ndarray):
        """Update slot states with advanced transition logic and real-time tracking"""
        if floor_name not in self.floors:
            return
        
        detections = self.detect_vehicles_in_frame(frame)
        slot_detections = self.match_cars_to_slots(detections, floor_name)
        
        self.current_frame += 1
        slot_detection_map = {d['slot_id']: d for d in slot_detections}
        
        for slot_id, slot in self.floors[floor_name].items():
            current_detection = slot_detection_map.get(slot_id)
            is_currently_occupied = bool(current_detection and current_detection['is_occupied'])
            
            slot.detection_history.append(is_currently_occupied)
            if len(slot.detection_history) > 5:
                slot.detection_history = slot.detection_history[-5:]
            
            slot.associated_car = current_detection['car'] if current_detection and is_currently_occupied else None
            slot.confidence = current_detection['overlap_ratio'] if current_detection else 0.0
            
            old_status = slot.status
            current_state = slot.status
            
            if current_state == "Free":
                if is_currently_occupied:
                    slot.update_status("Parking")
                    slot.transition_start_frame = self.current_frame
                    slot.stable = False
                    
            elif current_state == "Occupied":
                if not is_currently_occupied:
                    slot.update_status("Leaving")
                    slot.transition_start_frame = self.current_frame
                    slot.stable = False
                    
            elif current_state == "Parking":
                frames_in_transition = self.current_frame - (slot.transition_start_frame or self.current_frame)
                
                if frames_in_transition >= slot.transition_duration_frames:
                    recent_occupied = sum(slot.detection_history[-3:]) if len(slot.detection_history) >= 3 else 0
                    
                    if recent_occupied >= 2:
                        slot.update_status("Occupied")
                        slot.previous_stable_state = "Occupied"
                        slot.stable = True
                    else:
                        slot.update_status("Free")
                        slot.previous_stable_state = "Free"
                        slot.stable = True
                elif not is_currently_occupied and frames_in_transition > self.detection_params['fps'] * 2:
                    slot.update_status("Free")
                    slot.stable = True
                    
            elif current_state == "Leaving":
                frames_in_transition = self.current_frame - (slot.transition_start_frame or self.current_frame)
                
                if frames_in_transition >= slot.transition_duration_frames:
                    recent_occupied = sum(slot.detection_history[-3:]) if len(slot.detection_history) >= 3 else 0
                    
                    if recent_occupied <= 1:
                        slot.update_status("Free")
                        slot.previous_stable_state = "Free"
                        slot.stable = True
                    else:
                        slot.update_status("Occupied")
                        slot.previous_stable_state = "Occupied"
                        slot.stable = True
                elif is_currently_occupied and frames_in_transition > self.detection_params['fps'] * 2:
                    slot.update_status("Occupied")
                    slot.stable = True
            
            # Log status changes for real-time tracking
            if old_status != slot.status:
                event = {
                    'timestamp': datetime.now(),
                    'floor': floor_name,
                    'slot_id': slot_id,
                    'old_status': old_status,
                    'new_status': slot.status,
                    'confidence': slot.confidence
                }
                st.session_state.detection_log.append(event)
                
                # Update slot history
                history_key = f"{floor_name}_{slot_id}"
                st.session_state.slot_history[history_key].append({
                    'timestamp': datetime.now(),
                    'status': slot.status
                })
        
        # Log periodic statistics (every 30 frames)
        if self.current_frame % 30 == 0:
            self.log_realtime_stats()
    
    def log_realtime_stats(self):
        """Log real-time statistics for analytics"""
        now = datetime.now()
        
        stats_record = {
            'timestamp': now,
            'total_slots': 0,
            'free_slots': 0,
            'occupied_slots': 0,
            'parking_slots': 0,
            'leaving_slots': 0
        }
        
        for floor_name, slots in self.floors.items():
            floor_stats = self.get_floor_stats(floor_name)
            stats_record[f'{floor_name}_free'] = floor_stats['Free']
            stats_record[f'{floor_name}_occupied'] = floor_stats['Occupied']
            stats_record[f'{floor_name}_parking'] = floor_stats['Parking']
            stats_record[f'{floor_name}_leaving'] = floor_stats['Leaving']
            
            stats_record['total_slots'] += floor_stats['Total']
            stats_record['free_slots'] += floor_stats['Free']
            stats_record['occupied_slots'] += floor_stats['Occupied']
            stats_record['parking_slots'] += floor_stats['Parking']
            stats_record['leaving_slots'] += floor_stats['Leaving']
        
        st.session_state.historical_data.append(stats_record)
        st.session_state.occupancy_timeline.append({
            'timestamp': now,
            'occupancy_rate': (stats_record['occupied_slots'] / stats_record['total_slots'] * 100) if stats_record['total_slots'] > 0 else 0
        })
    
    def draw_slots_on_frame(self, frame: np.ndarray, floor_name: str) -> np.ndarray:
        """Draw parking slots on frame with enhanced visualization"""
        if floor_name not in self.floors:
            return frame
        
        overlay = frame.copy()
        slots = self.floors[floor_name]
        
        for slot in slots.values():
            color = slot.get_color()
            
            cv2.fillPoly(overlay, [slot.polygon], color)
            thickness = 4 if slot.status == "Occupied" else 3
            cv2.polylines(overlay, [slot.polygon], True, color, thickness)
            
            frames_remaining = 0
            if slot.transition_start_frame is not None:
                frames_passed = self.current_frame - slot.transition_start_frame
                frames_remaining = max(0, slot.transition_duration_frames - frames_passed)
            
            seconds_remaining = frames_remaining // self.detection_params['fps']
            
            if slot.status == "Parking":
                status_text = f"PARKING ({seconds_remaining}s)"
            elif slot.status == "Leaving":
                status_text = f"LEAVING ({seconds_remaining}s)"
            elif slot.status == "Occupied":
                status_text = "OCCUPIED"
            else:
                status_text = "FREE"
            
            center = slot.center
            lines = [
                f"{slot.id}",
                status_text,
                f"({slot.confidence:.2f})"
            ]
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness_text = 2
            
            for i, line in enumerate(lines):
                text_size = cv2.getTextSize(line, font, font_scale, thickness_text)[0]
                text_x = center[0] - text_size[0] // 2
                text_y = center[1] - 25 + i * 20
                
                cv2.rectangle(overlay,
                            (text_x - 3, text_y - 15),
                            (text_x + text_size[0] + 3, text_y + 3),
                            (0, 0, 0), -1)
                
                cv2.putText(overlay, line, (text_x, text_y),
                          font, font_scale, (255, 255, 255), thickness_text)
        
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, overlay)
        
        total_slots = len(slots)
        free_slots = sum(1 for s in slots.values() if s.status == 'Free')
        occupied_slots = sum(1 for s in slots.values() if s.status == 'Occupied')
        transition_slots = sum(1 for s in slots.values() if s.status in ['Parking', 'Leaving'])
        
        stats_height = 90
        cv2.rectangle(overlay, (10, 10), (600, stats_height), (0, 0, 0), -1)
        
        stats_lines = [
            f"{floor_name} - Frame: {self.frame_counts.get(floor_name, 0)}",
            f"Free: {free_slots} | Occupied: {occupied_slots} | Changing: {transition_slots}",
            f"Total Slots: {total_slots} | FPS: {self.detection_params['fps']}"
        ]
        
        for i, line in enumerate(stats_lines):
            cv2.putText(overlay, line, (15, 30 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return overlay
    
    def get_floor_stats(self, floor_name: str) -> Dict:
        """Get statistics for a floor"""
        if floor_name not in self.floors:
            return {"Free": 0, "Occupied": 0, "Parking": 0, "Leaving": 0, "Total": 0, "Occupancy_Rate": 0}
        
        slots = self.floors[floor_name]
        stats = {
            "Free": sum(1 for s in slots.values() if s.status == 'Free'),
            "Occupied": sum(1 for s in slots.values() if s.status == 'Occupied'),
            "Parking": sum(1 for s in slots.values() if s.status == 'Parking'),
            "Leaving": sum(1 for s in slots.values() if s.status == 'Leaving')
        }
        
        stats["Total"] = sum(stats.values())
        stats["Occupancy_Rate"] = (stats["Occupied"] / stats["Total"] * 100) if stats["Total"] > 0 else 0
        
        return stats
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all floors"""
        all_stats = {}
        for floor_name in self.floors:
            all_stats[floor_name] = self.get_floor_stats(floor_name)
        return all_stats
    
    def cleanup(self):
        """Release video resources"""
        for cap in self.video_captures.values():
            cap.release()
        self.video_captures.clear()

def create_sample_json_template(floor_type: str) -> dict:
    """Create sample JSON structure"""
    if floor_type == "ground":
        return {
            "floor_name": "Ground Floor",
            "slots": [
                {"id": "G1", "polygon": [[100, 200], [200, 200], [200, 280], [100, 280]]},
                {"id": "G2", "polygon": [[220, 200], [320, 200], [320, 280], [220, 280]]},
                {"id": "G3", "polygon": [[340, 200], [440, 200], [440, 280], [340, 280]]},
                {"id": "G4", "polygon": [[100, 300], [200, 300], [200, 380], [100, 380]]},
                {"id": "G5", "polygon": [[220, 300], [320, 300], [320, 380], [220, 380]]}
            ]
        }
    else:
        return {
            "floor_name": "First Floor",
            "slots": [
                {"id": "F1", "polygon": [[100, 200], [200, 200], [200, 280], [100, 280]]},
                {"id": "F2", "polygon": [[220, 200], [320, 200], [320, 280], [220, 280]]},
                {"id": "F3", "polygon": [[340, 200], [440, 200], [440, 280], [340, 280]]},
                {"id": "F4", "polygon": [[100, 300], [200, 300], [200, 380], [100, 380]]}
            ]
        }

# Page Functions
def show_header():
    """Display the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🚗 Multi-Floor Parking Detection System</h1>
        <p>Real-time parking space monitoring using AI-powered computer vision</p>
    </div>
    """, unsafe_allow_html=True)

def show_sidebar_status():
    """Show system status in sidebar"""
    st.sidebar.markdown("### System Status")
    
    if st.session_state.detection_active and st.session_state.detector:
        status_color = "🟢"
        status_text = "ACTIVE"
        
        all_stats = st.session_state.detector.get_all_stats()
        total_slots = sum(stats['Total'] for stats in all_stats.values())
        total_free = sum(stats['Free'] for stats in all_stats.values())
        total_occupied = sum(stats['Occupied'] for stats in all_stats.values())
        total_transition = sum(stats.get('Parking', 0) + stats.get('Leaving', 0) for stats in all_stats.values())
    else:
        status_color = "🔴"
        status_text = "INACTIVE"
        total_slots = total_free = total_occupied = total_transition = 0
    
    st.sidebar.markdown(f"""
    <div class="sidebar-status">
        <h4>{status_color} Detection Status: {status_text}</h4>
        <p><strong>Last Update:</strong> {st.session_state.last_update.strftime('%H:%M:%S')}</p>
        <p><strong>Total Slots:</strong> {total_slots}</p>
        <p><strong>Free Slots:</strong> {total_free}</p>
        <p><strong>Occupied Slots:</strong> {total_occupied}</p>
        <p><strong>Transitioning:</strong> {total_transition}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.detection_active:
        if st.sidebar.button("🛑 Emergency Stop", type="secondary"):
            st.session_state.detection_active = False
            if st.session_state.detector:
                st.session_state.detector.cleanup()
            st.rerun()

def configuration_page():
    """Configuration page with file uploads"""
    st.header("⚙️ System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📹 Video Files")
        
        st.markdown("**Ground Floor Video**")
        ground_video = st.file_uploader(
            "Upload parking1.mp4 or similar",
            type=['mp4', 'avi', 'mov'],
            key="ground_video"
        )
        if ground_video:
            st.success(f"✅ {ground_video.name} uploaded")
            st.session_state.config['ground_floor_video'] = ground_video
        
        st.markdown("**First Floor Video**")
        first_video = st.file_uploader(
            "Upload parking2.mp4 or similar",
            type=['mp4', 'avi', 'mov'],
            key="first_video"
        )
        if first_video:
            st.success(f"✅ {first_video.name} uploaded")
            st.session_state.config['first_floor_video'] = first_video
    
    with col2:
        st.subheader("📁 Slot Configuration Files")
        
        st.markdown("**Ground Floor Slots JSON**")
        ground_json = st.file_uploader(
            "Upload slots1.json",
            type=['json'],
            key="ground_json"
        )
        if ground_json:
            st.success(f"✅ {ground_json.name} uploaded")
            st.session_state.config['ground_floor_slots'] = ground_json
        
        if st.button("📄 Download Sample Ground Floor JSON"):
            sample_json = create_sample_json_template("ground")
            st.download_button(
                "Download JSON",
                data=json.dumps(sample_json, indent=2),
                file_name="ground_floor_slots.json",
                mime="application/json"
            )
        
        st.markdown("**First Floor Slots JSON**")
        first_json = st.file_uploader(
            "Upload slots2.json",
            type=['json'],
            key="first_json"
        )
        if first_json:
            st.success(f"✅ {first_json.name} uploaded")
            st.session_state.config['first_floor_slots'] = first_json
        
        if st.button("📄 Download Sample First Floor JSON"):
            sample_json = create_sample_json_template("first")
            st.download_button(
                "Download JSON",
                data=json.dumps(sample_json, indent=2),
                file_name="first_floor_slots.json",
                mime="application/json"
            )
    
    st.subheader("🤖 AI Model Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_threshold = st.slider(
            "Detection Confidence Threshold", 
            0.0, 1.0, 
            st.session_state.config['confidence_threshold'], 
            0.05
        )
        st.session_state.config['confidence_threshold'] = confidence_threshold
    
    with col2:
        transition_time = st.number_input(
            "Transition Time (seconds)", 
            min_value=1, max_value=60, 
            value=st.session_state.config['transition_time']
        )
        st.session_state.config['transition_time'] = transition_time
    
    st.subheader("🎬 System Control")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Detection", disabled=st.session_state.detection_active, type="primary"):
            if not all([
                st.session_state.config.get('ground_floor_video'),
                st.session_state.config.get('first_floor_video'),
                st.session_state.config.get('ground_floor_slots'),
                st.session_state.config.get('first_floor_slots')
            ]):
                st.error("❌ Please upload all required files before starting detection!")
            else:
                with st.spinner("Initializing detection system..."):
                    detector = EnhancedParkingDetectionSystem()
                    detector.load_model(confidence_threshold)
                    
                    detector.detection_params['transition_time_seconds'] = transition_time
                    detector.transition_frames = transition_time * detector.detection_params['fps']
                    
                    ground_video_loaded = detector.load_video(
                        st.session_state.config['ground_floor_video'], 
                        'Ground Floor'
                    )
                    first_video_loaded = detector.load_video(
                        st.session_state.config['first_floor_video'], 
                        'First Floor'
                    )
                    
                    ground_slots = detector.load_slots_from_json(
                        st.session_state.config['ground_floor_slots'], 
                        'Ground Floor'
                    )
                    first_slots = detector.load_slots_from_json(
                        st.session_state.config['first_floor_slots'], 
                        'First Floor'
                    )
                    
                    if ground_video_loaded and first_video_loaded and ground_slots > 0 and first_slots > 0:
                        st.session_state.detector = detector
                        st.session_state.detection_active = True
                        st.success(f"✅ System started! Ground Floor: {ground_slots} slots, First Floor: {first_slots} slots")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to initialize system. Check your files!")
    
    with col2:
        if st.button("⏹️ Stop Detection", disabled=not st.session_state.detection_active):
            st.session_state.detection_active = False
            if st.session_state.detector:
                st.session_state.detector.cleanup()
            st.success("Detection stopped!")
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset System"):
            st.session_state.detection_active = False
            if st.session_state.detector:
                st.session_state.detector.cleanup()
            st.session_state.detector = None
            st.session_state.historical_data.clear()
            st.session_state.slot_history.clear()
            st.session_state.occupancy_timeline.clear()
            st.session_state.detection_log.clear()
            st.success("System reset!")
            st.rerun()

def live_dashboard_page():
    """Live dashboard page with real-time updates"""
    st.header("📹 Live Dashboard")
    
    if not st.session_state.detection_active or not st.session_state.detector:
        st.warning("⚠️ Detection system is not active. Please start it from the Configuration page.")
        return
    
    detector = st.session_state.detector
    
    # Real-time statistics cards
    all_stats = detector.get_all_stats()
    total_slots = sum(stats['Total'] for stats in all_stats.values())
    total_free = sum(stats['Free'] for stats in all_stats.values())
    total_occupied = sum(stats['Occupied'] for stats in all_stats.values())
    total_transition = sum(stats.get('Parking', 0) + stats.get('Leaving', 0) for stats in all_stats.values())
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_slots}</h3>
            <p>Total Slots</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%);">
            <h3>{total_free}</h3>
            <p>Free Slots</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);">
            <h3>{total_occupied}</h3>
            <p>Occupied Slots</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);">
            <h3>{total_transition}</h3>
            <p>Transitioning</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Playback controls
    st.subheader("🎬 Playback Controls")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        playback_speed = st.slider("Playback Speed", 0.25, 3.0, 1.0, 0.25)
    
    with col2:
        frame_skip = st.selectbox("Frame Skip", [1, 2, 3, 5], index=0)
    
    with col3:
        max_frames = st.number_input("Max Frames", 100, 1000, 300, 50)
    
    # Live video full width
    st.subheader("🎥 Live Video Feeds")
    video_placeholder = st.empty()
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # Process and display video frames
    frame_count = 0
    display_frame_interval = max(1, frame_skip)
    
    try:
        while frame_count < max_frames and st.session_state.detection_active:
            frames_to_display = []
            
            for floor_name in ["First Floor", "Ground Floor"]:
                if floor_name in detector.video_captures:
                    frame = None
                    for _ in range(frame_skip):
                        frame = detector.get_video_frame(floor_name)
                        if frame is None:
                            break
                    
                    if frame is not None:
                        detector.update_floor_slots(floor_name, frame)
                        processed_frame = detector.draw_slots_on_frame(frame, floor_name)
                        
                        label_text = f"{floor_name.upper()} - LIVE DETECTION"
                        if detector.use_mock:
                            label_text += " (SIMULATION MODE)"
                        else:
                            label_text += " (YOLO AI)"
                        
                        cv2.putText(processed_frame, label_text, (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                        
                        frames_to_display.append(processed_frame)
            
            if len(frames_to_display) == 2:
                max_width = max(f.shape[1] for f in frames_to_display)
                resized = []
                for f in frames_to_display:
                    if f.shape[1] != max_width:
                        f = cv2.resize(f, (max_width, f.shape[0]))
                    resized.append(f)
                
                combined = np.vstack(resized)
            elif len(frames_to_display) == 1:
                combined = frames_to_display[0]
            else:
                st.error("❌ No frames available from videos")
                break
            
            if frame_count % display_frame_interval == 0:
                video_placeholder.image(combined, channels="BGR", use_container_width=True)
            
            progress = frame_count / max_frames
            progress_bar.progress(progress)
            
            # Refresh stats
            all_stats = detector.get_all_stats()
            total_free = sum(stats['Free'] for stats in all_stats.values())
            total_occupied = sum(stats['Occupied'] for stats in all_stats.values())
            total_transition = sum(stats.get('Parking', 0) + stats.get('Leaving', 0) for stats in all_stats.values())
            
            stats_summary = f"Free: {total_free} | Occupied: {total_occupied} | Changing: {total_transition}"
            status_text.text(f"🎬 Frame {frame_count}/{max_frames} | Speed: {playback_speed}x | {stats_summary}")
            
            st.session_state.last_update = datetime.now()
            
            base_delay = 0.033
            adjusted_delay = (base_delay / playback_speed) / frame_skip
            time.sleep(max(0.001, adjusted_delay))
            
            frame_count += 1
        
        progress_bar.progress(1.0)
        status_text.text("✅ Processing completed! Click 'Refresh Status' to continue.")
        
    except Exception as e:
        st.error(f"❌ Error during processing: {e}")
        import traceback
        st.error(traceback.format_exc())
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Status", key="refresh_live"):
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Detection"):
            st.session_state.detection_active = False
            detector.cleanup()
            st.rerun()

def analytics_page():
    """Real-time analytics page with live data"""
    st.header("📊 Real-Time Analytics Dashboard")
    
    if not st.session_state.detector or not st.session_state.detection_active:
        st.warning("⚠️ Detection must be active to view real-time analytics. Start detection from Configuration page.")
        
        # Show info about data collection
        if len(st.session_state.historical_data) > 0:
            st.info(f"📦 {len(st.session_state.historical_data)} historical data points available from previous session.")
        else:
            st.info("💡 Start detection to begin collecting real-time analytics data.")
            return
    
    # Check if we have data
    if len(st.session_state.historical_data) == 0:
        st.info("⏳ Collecting data... Please wait for detection to gather statistics.")
        return
    
    # Convert real-time data to DataFrame
    df = pd.DataFrame(list(st.session_state.historical_data))
    
    # Overall statistics from current session
    st.subheader("📈 Current Session Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_occupancy = df['occupied_slots'].mean()
        st.metric("Average Occupancy", f"{avg_occupancy:.1f} slots", 
                 f"{(avg_occupancy/df['total_slots'].iloc[-1])*100:.1f}%")
    
    with col2:
        peak_occupancy = df['occupied_slots'].max()
        st.metric("Peak Occupancy", f"{peak_occupancy} slots",
                 f"{(peak_occupancy/df['total_slots'].iloc[-1])*100:.1f}%")
    
    with col3:
        current_occupancy = df['occupied_slots'].iloc[-1] if not df.empty else 0
        st.metric("Current Occupancy", f"{current_occupancy} slots",
                 f"{(current_occupancy/df['total_slots'].iloc[-1])*100:.1f}%")
    
    with col4:
        session_duration = (datetime.now() - df['timestamp'].iloc[0]).total_seconds() / 60
        st.metric("Session Duration", f"{session_duration:.1f} min")
    
    # Real-time charts
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Real-Time Occupancy Trend', 'Floor-wise Distribution', 
                       'Status Distribution', 'Occupancy Rate Over Time'),
        specs=[[{"secondary_y": False}, {"type": "bar"}],
               [{"type": "pie"}, {"type": "scatter"}]]
    )
    
    # Occupancy trend
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['occupied_slots'],
                  mode='lines', name='Occupied', line=dict(color='red', width=2),
                  fill='tonexty'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['free_slots'],
                  mode='lines', name='Free', line=dict(color='green', width=2)),
        row=1, col=1
    )
    
    # Floor comparison (most recent data)
    latest_data = df.iloc[-1]
    floors_data = []
    for floor_name in ['Ground Floor', 'First Floor']:
        if f'{floor_name}_occupied' in df.columns:
            floors_data.append({
                'floor': floor_name,
                'occupied': latest_data[f'{floor_name}_occupied'],
                'free': latest_data[f'{floor_name}_free']
            })
    
    if floors_data:
        floors_df = pd.DataFrame(floors_data)
        fig.add_trace(
            go.Bar(x=floors_df['floor'], y=floors_df['occupied'],
                   name='Occupied', marker_color='red'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=floors_df['floor'], y=floors_df['free'],
                   name='Free', marker_color='green'),
            row=1, col=2
        )
    
    # Status distribution pie chart
    latest_data = df.iloc[-1]
    status_values = [
        latest_data['free_slots'],
        latest_data['occupied_slots'],
        latest_data.get('parking_slots', 0),
        latest_data.get('leaving_slots', 0)
    ]
    status_labels = ['Free', 'Occupied', 'Parking', 'Leaving']
    
    fig.add_trace(
        go.Pie(labels=status_labels, values=status_values,
               marker=dict(colors=['#28a745', '#dc3545', '#ffc107', '#fd7e14'])),
        row=2, col=1
    )
    
    # Occupancy rate timeline
    if len(st.session_state.occupancy_timeline) > 0:
        timeline_df = pd.DataFrame(list(st.session_state.occupancy_timeline))
        fig.add_trace(
            go.Scatter(x=timeline_df['timestamp'], y=timeline_df['occupancy_rate'],
                      mode='lines+markers', name='Occupancy %', 
                      line=dict(color='purple', width=2), marker=dict(size=4)),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=True, title_text="Live Parking Analytics")
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Number of Slots", row=1, col=1)
    fig.update_yaxes(title_text="Occupancy Rate (%)", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity timeline
    st.subheader("🕐 Recent Activity Timeline")
    
    if len(st.session_state.detection_log) > 0:
        recent_log = list(st.session_state.detection_log)[-20:]
        log_df = pd.DataFrame(recent_log)
        log_df['time'] = log_df['timestamp'].dt.strftime('%H:%M:%S')
        log_df = log_df[['time', 'floor', 'slot_id', 'old_status', 'new_status', 'confidence']]
        log_df.columns = ['Time', 'Floor', 'Slot', 'From', 'To', 'Confidence']
        
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.info("No status changes recorded yet.")
    
    # Export real-time data
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Session Data"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"parking_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("🔄 Refresh Analytics"):
            st.rerun()

def detailed_slots_page():
    """Real-time detailed slots information page"""
    st.header("🅿️ Detailed Slot Information (Live)")
    
    if not st.session_state.detector or not st.session_state.detection_active:
        st.warning("⚠️ No detection data available. Start detection from Configuration page.")
        return
    
    detector = st.session_state.detector
    
    # Auto-refresh control
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 🔴 LIVE - Updates in real-time")
    with col2:
        if st.button("🔄 Manual Refresh"):
            st.rerun()
    
    # Floor filter
    floor_options = ["All"] + list(detector.floors.keys())
    floor_filter = st.selectbox("Select Floor", floor_options)
    
    # Collect real-time slot data
    slots_data = []
    
    for floor_name, floor_slots in detector.floors.items():
        if floor_filter != "All" and floor_filter != floor_name:
            continue
        
        for slot_id, slot in floor_slots.items():
            frames_remaining = 0
            if slot.transition_start_frame is not None:
                frames_passed = detector.current_frame - slot.transition_start_frame
                frames_remaining = max(0, slot.transition_duration_frames - frames_passed)
            
            seconds_remaining = frames_remaining // detector.detection_params['fps']
            
            # Calculate how long in current status
            time_in_status = (datetime.now() - slot.last_status_change).total_seconds()
            
            slots_data.append({
                'Slot ID': slot.id,
                'Floor': floor_name,
                'Status': slot.status,
                'Confidence': f"{slot.confidence:.2f}",
                'Transition Timer': f"{seconds_remaining}s" if seconds_remaining > 0 else "-",
                'Time in Status': f"{int(time_in_status)}s",
                'Total Occupations': slot.total_occupations,
                'Stable': "✓" if slot.stable else "✗"
            })
    
    if not slots_data:
        st.info("No slot data available.")
        return
    
    df_slots = pd.DataFrame(slots_data)
    
    # Status filter
    status_filter = st.multiselect(
        "Filter by Status",
        options=['Free', 'Occupied', 'Parking', 'Leaving'],
        default=['Free', 'Occupied', 'Parking', 'Leaving']
    )
    
    if status_filter:
        df_slots = df_slots[df_slots['Status'].isin(status_filter)]
    
    # Real-time statistics summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_slots = len(slots_data)
        st.metric("Total Slots", total_slots)
    
    with col2:
        free_slots = len([s for s in slots_data if s['Status'] == 'Free'])
        st.metric("Free Slots", free_slots, f"{(free_slots/total_slots*100):.1f}%" if total_slots > 0 else "0%")
    
    with col3:
        occupied_slots = len([s for s in slots_data if s['Status'] == 'Occupied'])
        st.metric("Occupied Slots", occupied_slots, f"{(occupied_slots/total_slots*100):.1f}%" if total_slots > 0 else "0%")
    
    with col4:
        transition_slots = len([s for s in slots_data if s['Status'] in ['Parking', 'Leaving']])
        st.metric("Transitioning", transition_slots, f"{(transition_slots/total_slots*100):.1f}%" if total_slots > 0 else "0%")
    
    # Slots table with color coding
    st.subheader("Real-Time Slot Status")
    
    def color_status(val):
        if val == 'Free':
            return 'background-color: #d4edda; color: #155724'
        elif val == 'Occupied':
            return 'background-color: #f8d7da; color: #721c24'
        elif val in ['Parking', 'Leaving']:
            return 'background-color: #fff3cd; color: #856404'
        return ''
    
    styled_df = df_slots.style.applymap(color_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Slot history visualization
    st.subheader("📊 Slot Activity History")
    
    if df_slots.empty:
        st.info("No slots to display.")
    else:
        # Create slot selector
        slot_ids = df_slots['Slot ID'].tolist()
        selected_slot = st.selectbox("Select Slot to View History", slot_ids)
        
        if selected_slot:
            # Find the slot in detector
            slot_obj = None
            slot_floor = None
            for floor_name, floor_slots in detector.floors.items():
                if selected_slot in floor_slots:
                    slot_obj = floor_slots[selected_slot]
                    slot_floor = floor_name
                    break
            
            if slot_obj:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Status", slot_obj.status)
                with col2:
                    st.metric("Confidence", f"{slot_obj.confidence:.2%}")
                with col3:
                    st.metric("Total Uses", slot_obj.total_occupations)
                
                # Show history if available
                history_key = f"{slot_floor}_{selected_slot}"
                if history_key in st.session_state.slot_history and len(st.session_state.slot_history[history_key]) > 0:
                    history = st.session_state.slot_history[history_key]
                    history_df = pd.DataFrame(history)
                    history_df['time'] = history_df['timestamp'].dt.strftime('%H:%M:%S')
                    
                    # Create timeline visualization
                    fig = go.Figure()
                    
                    status_colors = {
                        'Free': 'green',
                        'Occupied': 'red',
                        'Parking': 'orange',
                        'Leaving': 'yellow'
                    }
                    
                    for idx, row in history_df.iterrows():
                        fig.add_trace(go.Scatter(
                            x=[row['timestamp']],
                            y=[row['status']],
                            mode='markers',
                            marker=dict(size=15, color=status_colors.get(row['status'], 'gray')),
                            name=row['status'],
                            showlegend=False
                        ))
                    
                    fig.update_layout(
                        title=f"Status History for {selected_slot}",
                        xaxis_title="Time",
                        yaxis_title="Status",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show history table
                    st.dataframe(history_df[['time', 'status']], use_container_width=True, hide_index=True)
                else:
                    st.info("No history recorded yet for this slot.")
    
    # Export functionality
    if st.button("📊 Export Slot Data"):
        csv = df_slots.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"parking_slots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def setup_guide_page():
    """Setup guide and documentation"""
    st.header("📖 Setup Guide")
    
    st.markdown("""
    ## 🚀 Quick Start Guide
    
    ### Step 1: Prepare Your Files
    
    **📹 Video Files:**
    - Ground floor parking video (MP4, AVI, or MOV)
    - First floor parking video (MP4, AVI, or MOV)
    
    **📁 JSON Slot Files:**
    Your JSON can be in this simple format:
    ```json
    [
      [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    ]
    ```
    
    ### Step 2: Upload Files
    1. Go to **Configuration** page
    2. Upload both parking videos
    3. Upload both JSON slot files
    4. Adjust AI settings if needed
    5. Click **Start Detection**
    
    ### Step 3: Monitor Live
    - Go to **Live Dashboard** for real-time video feeds
    - Check **Analytics** for live statistics and charts
    - View **Detailed Slots** for individual slot tracking
    - All pages update with real-time data automatically!
    
    ## 🎬 Real-Time Features
    
    **Live Dashboard:**
    - Real-time video processing
    - Live event feed showing status changes
    - Adjustable playback speed (0.25x - 3.0x)
    - Frame skip for performance optimization
    
    **Analytics:**
    - Live occupancy trends
    - Real-time floor comparisons
    - Session statistics
    - Exportable historical data
    
    **Detailed Slots:**
    - Individual slot monitoring
    - Status change history
    - Confidence scores
    - Occupation counters
    
    ## 🎯 Detection Features
    
    - ✅ Real-time data collection and storage
    - ✅ Live event logging (last 100 events)
    - ✅ Historical data tracking (last 1000 records)
    - ✅ Individual slot history
    - ✅ Advanced overlap calculation (3 methods)
    - ✅ 10-second transition smoothing
    - ✅ False positive prevention
    - ✅ Confidence scoring
    - ✅ YOLO or simulation mode
    
    ## 📊 Status States
    
    - **🟢 FREE**: Parking slot is available
    - **🔴 OCCUPIED**: Vehicle is parked
    - **🟠 PARKING**: Vehicle entering (10s countdown)
    - **🟡 LEAVING**: Vehicle exiting (10s countdown)
    
    ## 💡 Tips for Best Results
    
    1. Use good quality videos (720p+)
    2. Ensure proper lighting in videos
    3. Mark slot boundaries accurately
    4. Use confidence threshold 0.25-0.5
    5. Keep transition time at 10 seconds
    6. Monitor Analytics page for trends
    7. Export data regularly for analysis
    
    ## 🔄 Real-Time Data Flow
    
    ```
    Video Frame → Vehicle Detection → Slot Matching → Status Update
                                                            ↓
    Live Dashboard ← Analytics ← Historical Data ← Status Change Log
    ```
    
    All three monitoring pages (Live Dashboard, Analytics, Detailed Slots) 
    receive real-time updates as the detection system processes video frames.
    
    ## 📈 Data Persistence
    
    - Session data is stored in memory during detection
    - Historical data: Last 1000 records
    - Event log: Last 100 status changes
    - Slot history: Complete timeline per slot
    - Export data before resetting the system
    
    ## ⚡ Performance Tips
    
    - Use frame skip (2-5) for faster processing
    - Higher playback speed reduces detail but increases throughput
    - Monitor system resources if processing large videos
    - Stop detection when not needed to save resources
    """)

def main():
    """Main application function"""
    initialize_session_state()
    
    show_header()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    pages = {
        "⚙️ Configuration": configuration_page,
        "📹 Live Dashboard": live_dashboard_page,
        "📊 Analytics": analytics_page,
        "🅿️ Detailed Slots": detailed_slots_page,
        "📖 Setup Guide": setup_guide_page
    }
    
    selected_page = st.sidebar.selectbox("Select Page", list(pages.keys()))
    
    # Show sidebar status
    show_sidebar_status()
    
    # Show data collection info in sidebar
    if st.session_state.detection_active:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Data Collection")
        st.sidebar.info(f"""
        **Historical Records:** {len(st.session_state.historical_data)}  
        **Event Log:** {len(st.session_state.detection_log)}  
        **Tracked Slots:** {len(st.session_state.slot_history)}
        """)
    
    # Display selected page
    pages[selected_page]()
    
    # Auto-refresh for live pages (optional - can be commented out if not needed)
    if selected_page in ["📹 Live Dashboard", "📊 Analytics", "🅿️ Detailed Slots"]:
        if st.session_state.detection_active:
            st.markdown("""
            <style>
            .stApp {
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.95; }
            }
            </style>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
