import cv2
import json
import numpy as np
import pandas as pd
from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
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
import streamlit as st

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
    }
    .status-free { color: #28a745; font-weight: bold; }
    .status-occupied { color: #dc3545; font-weight: bold; }
    .status-transitioning { color: #ffc107; font-weight: bold; }
    .floor-header {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin: 1rem 0;
    }
    .integrated-mode {
        background: linear-gradient(45deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# INTEGRATED VIDEO PATHS - Replace these with your actual video file paths
INTEGRATED_VIDEOS = {
    "ground_floor": "D:\Engineering\Projects\Smart Parking\parking1.mp4",  # Put your ground floor video here
    "first_floor": "D:\Engineering\Projects\Smart Parking\parking2.mp4"     # Put your first floor video here
}

# Alternative: Use relative paths from script location
# INTEGRATED_VIDEOS = {
#     "ground_floor": os.path.join(os.path.dirname(__file__), "videos", "ground_floor_parking.mp4"),
#     "first_floor": os.path.join(os.path.dirname(__file__), "videos", "first_floor_parking.mp4")
# }

# Embedded slot configurations
GROUND_FLOOR_SLOTS = [
    {"id": "G1", "polygon": [[499, 1056], [79, 1057], [75, 810], [449, 817]]},
    {"id": "G2", "polygon": [[419, 712], [180, 711], [178, 686], [409, 671]]},
    {"id": "G3", "polygon": [[532, 744], [486, 683], [711, 676], [733, 713]]},
    {"id": "G4", "polygon": [[1905, 936], [1578, 840], [1753, 649], [1905, 661]]},
    {"id": "G5", "polygon": [[1890, 1015], [1515, 858], [1379, 886], [1671, 1038]]},
    {"id": "G6", "polygon": [[1419, 1053], [1253, 911], [1007, 937], [1122, 1051]]},
    {"id": "G7", "polygon": [[830, 677], [854, 721], [966, 716], [929, 667]]},
    {"id": "G8", "polygon": [[591, 943], [636, 1057], [911, 1062], [837, 941]]}
]

FIRST_FLOOR_SLOTS = [
    {"id": "F1", "polygon": [[1760, 950], [1510, 975], [1380, 899], [1671, 893]]},
    {"id": "F2", "polygon": [[1648, 874], [1355, 888], [1213, 849], [1520, 806]]},
    {"id": "F3", "polygon": [[1174, 836], [1056, 804], [959, 817], [1056, 867]]},
    {"id": "F4", "polygon": [[985, 795], [905, 681], [812, 757], [885, 810]]},
    {"id": "F5", "polygon": [[895, 679], [820, 652], [711, 734], [780, 781]]},
    {"id": "F6", "polygon": [[761, 665], [714, 644], [622, 703], [687, 755]]},
    {"id": "F7", "polygon": [[640, 676], [478, 672], [495, 601], [658, 599]]}
]

class ParkingSlot:
    def __init__(self, slot_id: str, polygon: List[List[int]]):
        self.id = slot_id
        self.polygon = np.array(polygon, dtype=np.int32)
        self.status = "Free"
        self.confidence = 0.0
        self.transition_start = None
        self.transition_duration = 10  # seconds
        self.status_history = []
        
    def update_detection(self, detected: bool, confidence: float):
        current_time = time.time()
        
        if detected and self.status == "Free":
            if self.transition_start is None:
                self.transition_start = current_time
                self.status = "Parking"
            elif current_time - self.transition_start >= self.transition_duration:
                self.status = "Occupied"
                self.confidence = confidence
                self.transition_start = None
                self._add_to_history("Occupied")
                
        elif not detected and self.status == "Occupied":
            if self.transition_start is None:
                self.transition_start = current_time
                self.status = "Leaving"
            elif current_time - self.transition_start >= self.transition_duration:
                self.status = "Free"
                self.confidence = 0.0
                self.transition_start = None
                self._add_to_history("Free")
                
        elif detected and self.status in ["Parking", "Occupied"]:
            self.confidence = max(self.confidence, confidence)
            if self.status == "Parking" and self.transition_start:
                remaining = self.transition_duration - (current_time - self.transition_start)
                if remaining <= 0:
                    self.status = "Occupied"
                    self.transition_start = None
                    self._add_to_history("Occupied")
                    
        elif not detected and self.status in ["Leaving", "Free"]:
            if self.status == "Leaving" and self.transition_start:
                remaining = self.transition_duration - (current_time - self.transition_start)
                if remaining <= 0:
                    self.status = "Free"
                    self.confidence = 0.0
                    self.transition_start = None
                    self._add_to_history("Free")
    
    def _add_to_history(self, status: str):
        self.status_history.append({
            'timestamp': datetime.now(),
            'status': status,
            'confidence': self.confidence
        })
        if len(self.status_history) > 100:
            self.status_history = self.status_history[-100:]
    
    def get_remaining_transition_time(self) -> float:
        if self.transition_start is None:
            return 0
        return max(0, self.transition_duration - (time.time() - self.transition_start))
    
    def get_color(self) -> Tuple[int, int, int]:
        colors = {
            "Free": (0, 255, 0),
            "Occupied": (0, 0, 255),
            "Parking": (0, 165, 255),
            "Leaving": (255, 165, 0)
        }
        return colors.get(self.status, (128, 128, 128))

class IntegratedVideoManager:
    """Manages integrated video files for the dashboard"""
    
    def __init__(self):
        self.ground_floor_cap = None
        self.first_floor_cap = None
        self.videos_loaded = False
        self.current_frame = {"ground": 0, "first": 0}
        
    def check_integrated_videos(self):
        """Check if integrated videos exist and are accessible"""
        ground_exists = os.path.exists(INTEGRATED_VIDEOS["ground_floor"])
        first_exists = os.path.exists(INTEGRATED_VIDEOS["first_floor"])
        
        return {
            "ground_floor": ground_exists,
            "first_floor": first_exists,
            "both_available": ground_exists and first_exists,
            "ground_path": INTEGRATED_VIDEOS["ground_floor"] if ground_exists else None,
            "first_path": INTEGRATED_VIDEOS["first_floor"] if first_exists else None
        }
    
    def load_integrated_videos(self):
        """Load the integrated video files"""
        video_status = self.check_integrated_videos()
        
        if video_status["ground_floor"]:
            self.ground_floor_cap = cv2.VideoCapture(INTEGRATED_VIDEOS["ground_floor"])
            if not self.ground_floor_cap.isOpened():
                st.error(f"Cannot open ground floor video: {INTEGRATED_VIDEOS['ground_floor']}")
                self.ground_floor_cap = None
        
        if video_status["first_floor"]:
            self.first_floor_cap = cv2.VideoCapture(INTEGRATED_VIDEOS["first_floor"])
            if not self.first_floor_cap.isOpened():
                st.error(f"Cannot open first floor video: {INTEGRATED_VIDEOS['first_floor']}")
                self.first_floor_cap = None
        
        self.videos_loaded = (self.ground_floor_cap is not None) or (self.first_floor_cap is not None)
        return self.videos_loaded
    
    def get_frame(self, floor_type: str) -> Optional[np.ndarray]:
        """Get frame from the specified floor video"""
        if floor_type == "ground" and self.ground_floor_cap:
            ret, frame = self.ground_floor_cap.read()
            if ret:
                self.current_frame["ground"] += 1
                return frame
            else:
                # Loop video
                self.ground_floor_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame["ground"] = 0
                ret, frame = self.ground_floor_cap.read()
                return frame if ret else None
                
        elif floor_type == "first" and self.first_floor_cap:
            ret, frame = self.first_floor_cap.read()
            if ret:
                self.current_frame["first"] += 1
                return frame
            else:
                # Loop video
                self.first_floor_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame["first"] = 0
                ret, frame = self.first_floor_cap.read()
                return frame if ret else None
        
        return None
    
    def get_video_info(self):
        """Get information about loaded videos"""
        info = {"ground_floor": {}, "first_floor": {}}
        
        if self.ground_floor_cap:
            info["ground_floor"] = {
                "fps": self.ground_floor_cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(self.ground_floor_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(self.ground_floor_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.ground_floor_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "current_frame": self.current_frame["ground"]
            }
        
        if self.first_floor_cap:
            info["first_floor"] = {
                "fps": self.first_floor_cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(self.first_floor_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(self.first_floor_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.first_floor_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "current_frame": self.current_frame["first"]
            }
        
        return info
    
    def create_synthetic_frame(self, floor_name: str, width: int = 1920, height: int = 1080) -> np.ndarray:
        """Create synthetic parking lot frame as fallback"""
        frame = np.ones((height, width, 3), dtype=np.uint8) * 80
        
        # Add parking lot markings
        cv2.line(frame, (0, height//2), (width, height//2), (255, 255, 255), 3)
        cv2.line(frame, (width//2, 0), (width//2, height), (255, 255, 255), 3)
        
        # Add floor label
        cv2.putText(frame, f"{floor_name}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Add noise for texture
        noise = np.random.randint(0, 30, (height, width, 3))
        frame = cv2.add(frame, noise.astype(np.uint8))
        
        return frame
    
    def close_videos(self):
        """Release video capture objects"""
        if self.ground_floor_cap:
            self.ground_floor_cap.release()
        if self.first_floor_cap:
            self.first_floor_cap.release()

class EnhancedParkingDetectionSystem:
    def __init__(self):
        self.model = None
        self.floors = {}
        self.processing = False
        self.stats_history = []
        self.video_manager = IntegratedVideoManager()
        
    def load_model(self, model_path: str = None):
        try:
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                self.model = YOLO('yolov8n.pt')
            return True
        except Exception as e:
            st.warning(f"YOLO model not available: {e}. Using simulation mode.")
            return False
            
    def load_default_slots(self):
        """Load the embedded slot configurations"""
        self.load_slots("Ground Floor", GROUND_FLOOR_SLOTS)
        self.load_slots("1st Floor", FIRST_FLOOR_SLOTS)
        
    def load_slots(self, floor_name: str, slots_data: List[dict]):
        slots = {}
        for slot_data in slots_data:
            slot_id = slot_data['id']
            polygon = slot_data['polygon']
            slots[slot_id] = ParkingSlot(slot_id, polygon)
        self.floors[floor_name] = slots
        
    def detect_vehicles_in_frame(self, frame: np.ndarray) -> List[dict]:
        if self.model is None:
            return self._simulate_vehicle_detections(frame)
            
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    vehicle_classes = [2, 3, 5, 7, 1]
                    if class_id in vehicle_classes and confidence > 0.3:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        detections.append({
                            'center': (center_x, center_y),
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': confidence,
                            'class_id': class_id
                        })
        
        return detections
    
    def _simulate_vehicle_detections(self, frame: np.ndarray) -> List[dict]:
        """Simulate vehicle detections"""
        detections = []
        height, width = frame.shape[:2]
        
        for _ in range(np.random.randint(3, 8)):
            center_x = np.random.randint(100, width - 100)
            center_y = np.random.randint(100, height - 100)
            confidence = np.random.uniform(0.5, 0.95)
            
            bbox = (center_x - 40, center_y - 20, center_x + 40, center_y + 20)
            
            detections.append({
                'center': (center_x, center_y),
                'bbox': bbox,
                'confidence': confidence,
                'class_id': 2
            })
        
        return detections
    
    def update_floor_slots(self, floor_name: str, frame: np.ndarray):
        if floor_name not in self.floors:
            return
            
        detections = self.detect_vehicles_in_frame(frame)
        
        for slot in self.floors[floor_name].values():
            detected = False
            max_confidence = 0.0
            
            for detection in detections:
                center_x, center_y = detection['center']
                if cv2.pointPolygonTest(slot.polygon, (center_x, center_y), False) >= 0:
                    detected = True
                    max_confidence = max(max_confidence, detection['confidence'])
            
            slot.update_detection(detected, max_confidence)
    
    def draw_slots_on_frame(self, frame: np.ndarray, floor_name: str) -> np.ndarray:
        if floor_name not in self.floors:
            return frame
            
        overlay = frame.copy()
        
        for slot in self.floors[floor_name].values():
            color = slot.get_color()
            cv2.fillPoly(overlay, [slot.polygon], color)
            
            center = np.mean(slot.polygon, axis=0).astype(int)
            cv2.putText(overlay, f"{slot.id}", 
                       (center[0]-15, center[1]-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(overlay, slot.status, 
                       (center[0]-25, center[1]+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            remaining = slot.get_remaining_transition_time()
            if remaining > 0:
                cv2.putText(overlay, f"{remaining:.1f}s", 
                           (center[0]-15, center[1]+30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, overlay)
        return overlay
    
    def get_floor_stats(self, floor_name: str) -> Dict[str, int]:
        if floor_name not in self.floors:
            return {"Free": 0, "Occupied": 0, "Parking": 0, "Leaving": 0, "Total": 0}
            
        stats = {"Free": 0, "Occupied": 0, "Parking": 0, "Leaving": 0}
        
        for slot in self.floors[floor_name].values():
            stats[slot.status] += 1
            
        stats["Total"] = sum(stats.values())
        stats["Occupancy_Rate"] = (stats["Occupied"] / stats["Total"] * 100) if stats["Total"] > 0 else 0
        
        return stats
    
    def get_all_stats(self) -> Dict[str, Dict[str, int]]:
        all_stats = {}
        for floor_name in self.floors:
            all_stats[floor_name] = self.get_floor_stats(floor_name)
        return all_stats

# Initialize session state
if 'detection_system' not in st.session_state:
    st.session_state.detection_system = EnhancedParkingDetectionSystem()
    st.session_state.processing_active = False
    st.session_state.stats_data = []
    
    # Load default slots
    st.session_state.detection_system.load_default_slots()

# Main header
st.markdown("""
<div class="main-header">
    <h1>🚗 Smart Multi-Floor Parking Detection System</h1>
    <p>Integrated Dashboard • Built-in Videos • Real-time AI Detection</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🔧 System Configuration")

# Check video availability
video_status = st.session_state.detection_system.video_manager.check_integrated_videos()

# Video status display
st.sidebar.subheader("📹 Integrated Videos")
if video_status["both_available"]:
    st.sidebar.markdown("""
    <div class="integrated-mode">
    <h4>✅ All Videos Available</h4>
    <p>Ground Floor: Ready</p>
    <p>1st Floor: Ready</p>
    </div>
    """, unsafe_allow_html=True)
elif video_status["ground_floor"] or video_status["first_floor"]:
    available = "Ground Floor" if video_status["ground_floor"] else "1st Floor"
    missing = "1st Floor" if video_status["ground_floor"] else "Ground Floor"
    st.sidebar.warning(f"⚠️ {available}: Available\n❌ {missing}: Missing")
else:
    st.sidebar.error("❌ No integrated videos found. Using simulation mode.")

# Display video paths for reference
with st.sidebar.expander("Video File Paths"):
    st.write(f"Ground Floor: `{INTEGRATED_VIDEOS['ground_floor']}`")
    st.write(f"1st Floor: `{INTEGRATED_VIDEOS['first_floor']}`")
    
    if not video_status["both_available"]:
        st.warning("Place your video files at the above paths to enable integrated video playback.")

# Model configuration
st.sidebar.subheader("🤖 AI Model")
if 'model_loaded' not in st.session_state:
    model_loaded = st.session_state.detection_system.load_model()
    st.session_state.model_loaded = model_loaded
    
    if model_loaded:
        st.sidebar.success("✅ YOLO model loaded")
    else:
        st.sidebar.info("🎯 Simulation mode active")

# Processing settings
st.sidebar.subheader("⚙️ Processing Settings")
detection_mode = st.sidebar.selectbox(
    "Detection Mode:",
    ["Integrated Videos (Recommended)", "Simulation Mode"]
)
processing_speed = st.sidebar.slider("Processing Speed", 0.5, 3.0, 1.0, 0.1)

# System status and controls
st.sidebar.subheader("🔄 System Control")

system_ready = len(st.session_state.detection_system.floors) >= 2

if system_ready:
    st.sidebar.success("🟢 System Ready")
    
    if st.sidebar.button("▶️ Start Detection", disabled=st.session_state.processing_active):
        st.session_state.processing_active = True
        st.rerun()
        
    if st.sidebar.button("⏹️ Stop Detection", disabled=not st.session_state.processing_active):
        st.session_state.processing_active = False
        st.rerun()
        
    if st.sidebar.button("🔄 Reset System"):
        st.session_state.processing_active = False
        for floor_slots in st.session_state.detection_system.floors.values():
            for slot in floor_slots.values():
                slot.status = "Free"
                slot.confidence = 0.0
                slot.transition_start = None
        st.rerun()
else:
    st.sidebar.warning("🟡 System not ready")

# Main content area
col1, col2 = st.columns([2.5, 1.5])

with col1:
    st.subheader("🎬 Live Detection Feed")
    
    if st.session_state.processing_active:
        video_container = st.container()
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
        with video_container:
            video_placeholder = st.empty()
        
        try:
            # Load integrated videos
            videos_loaded = st.session_state.detection_system.video_manager.load_integrated_videos()
            
            if videos_loaded and detection_mode == "Integrated Videos (Recommended)":
                status_text.text("🎬 Processing integrated parking videos with AI detection...")
                
                max_frames = 300
                frame_count = 0
                
                while frame_count < max_frames and st.session_state.processing_active:
                    # Get frames from integrated videos
                    frame_first = st.session_state.detection_system.video_manager.get_frame("first")
                    frame_ground = st.session_state.detection_system.video_manager.get_frame("ground")
                    
                    # Use synthetic frames as fallback
                    if frame_first is None:
                        frame_first = st.session_state.detection_system.video_manager.create_synthetic_frame("1st Floor")
                    if frame_ground is None:
                        frame_ground = st.session_state.detection_system.video_manager.create_synthetic_frame("Ground Floor")
                    
                    # Process frames with AI detection
                    st.session_state.detection_system.update_floor_slots("1st Floor", frame_first)
                    st.session_state.detection_system.update_floor_slots("Ground Floor", frame_ground)
                    
                    # Draw slots on frames
                    processed_frame_first = st.session_state.detection_system.draw_slots_on_frame(
                        frame_first, "1st Floor"
                    )
                    processed_frame_ground = st.session_state.detection_system.draw_slots_on_frame(
                        frame_ground, "Ground Floor"
                    )
                    
                    # Ensure same width for stacking
                    target_width = max(processed_frame_first.shape[1], processed_frame_ground.shape[1])
                    if processed_frame_first.shape[1] != target_width:
                        processed_frame_first = cv2.resize(processed_frame_first, (target_width, processed_frame_first.shape[0]))
                    if processed_frame_ground.shape[1] != target_width:
                        processed_frame_ground = cv2.resize(processed_frame_ground, (target_width, processed_frame_ground.shape[0]))
                    
                    # Add floor labels with video indicators
                    first_indicator = "🎬 INTEGRATED VIDEO" if st.session_state.detection_system.video_manager.first_floor_cap else "🎯 SIMULATION"
                    ground_indicator = "🎬 INTEGRATED VIDEO" if st.session_state.detection_system.video_manager.ground_floor_cap else "🎯 SIMULATION"
                    
                    cv2.putText(processed_frame_first, f"1ST FLOOR - {first_indicator}", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                    cv2.putText(processed_frame_ground, f"GROUND FLOOR - {ground_indicator}", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                    
                    # Stack frames: 1st Floor on TOP, Ground Floor on BOTTOM
                    combined_frame = np.vstack((processed_frame_first, processed_frame_ground))
                    
                    # Display combined frame
                    video_placeholder.image(combined_frame, channels="BGR", use_container_width=True)
                    
                    # Update progress
                    progress = frame_count / max_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Frame {frame_count}/{max_frames} - AI Detection Active")
                    
                    # Control processing speed
                    time.sleep(0.5 / processing_speed)
                    frame_count += 1
                
                # Clean up
                st.session_state.detection_system.video_manager.close_videos()
                status_text.text("✅ Detection completed! Click 'Start Detection' to run again.")
            
            else:
                # Simulation mode
                status_text.text("🎯 Running in simulation mode...")
                
                max_frames = 200
                frame_count = 0
                
                while frame_count < max_frames and st.session_state.processing_active:
                    # Create synthetic frames
                    frame_first = st.session_state.detection_system.video_manager.create_synthetic_frame("1st Floor")
                    frame_ground = st.session_state.detection_system.video_manager.create_synthetic_frame("Ground Floor")
                    
                    # Add random vehicles for simulation
                    if np.random.random() < 0.7:  # 70% chance to add vehicles
                        for slots, frame in [(FIRST_FLOOR_SLOTS, frame_first), (GROUND_FLOOR_SLOTS, frame_ground)]:
                            for slot_data in slots:
                                if np.random.random() < 0.4:  # 40% occupancy rate
                                    polygon = np.array(slot_data['polygon'], dtype=np.int32)
                                    center = np.mean(polygon, axis=0).astype(int)
                                    
                                    # Draw vehicle rectangle
                                    top_left = (center[0] - 40, center[1] - 20)
                                    bottom_right = (center[0] + 40, center[1] + 20)
                                    
                                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
                                    color = colors[np.random.randint(0, len(colors))]
                                    
                                    cv2.rectangle(frame, top_left, bottom_right, color, -1)
                                    cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 2)
                    
                    # Process frames with AI detection
                    st.session_state.detection_system.update_floor_slots("1st Floor", frame_first)
                    st.session_state.detection_system.update_floor_slots("Ground Floor", frame_ground)
                    
                    # Draw slots on frames
                    processed_frame_first = st.session_state.detection_system.draw_slots_on_frame(
                        frame_first, "1st Floor"
                    )
                    processed_frame_ground = st.session_state.detection_system.draw_slots_on_frame(
                        frame_ground, "Ground Floor"
                    )
                    
                    # Ensure same width
                    target_width = max(processed_frame_first.shape[1], processed_frame_ground.shape[1])
                    if processed_frame_first.shape[1] != target_width:
                        processed_frame_first = cv2.resize(processed_frame_first, (target_width, processed_frame_first.shape[0]))
                    if processed_frame_ground.shape[1] != target_width:
                        processed_frame_ground = cv2.resize(processed_frame_ground, (target_width, processed_frame_ground.shape[0]))
                    
                    # Add floor labels
                    cv2.putText(processed_frame_first, "1ST FLOOR - SIMULATION", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                    cv2.putText(processed_frame_ground, "GROUND FLOOR - SIMULATION", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                    
                    # Stack frames
                    combined_frame = np.vstack((processed_frame_first, processed_frame_ground))
                    video_placeholder.image(combined_frame, channels="BGR", use_container_width=True)
                    
                    # Update progress
                    progress = frame_count / max_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Simulation Frame {frame_count}/{max_frames}")
                    
                    time.sleep(0.8 / processing_speed)
                    frame_count += 1
                
                status_text.text("✅ Simulation completed!")
                
        except Exception as e:
            st.error(f"Processing error: {e}")
            st.session_state.processing_active = False
        
    else:
        # Show system overview when not processing
        st.markdown("""
        <div class="integrated-mode">
        <h3>🚗 Integrated Dashboard Ready</h3>
        <p>Click 'Start Detection' to begin real-time parking analysis!</p>
        <p><b>Features:</b></p>
        <ul>
        <li>✅ Built-in video integration</li>
        <li>✅ Real-time AI vehicle detection</li>
        <li>✅ Multi-floor visualization</li>
        <li>✅ Smart status transitions</li>
        <li>✅ Live analytics and statistics</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Show slot preview
        st.subheader("📍 Configured Parking Slots")
        
        col_preview1, col_preview2 = st.columns(2)
        
        with col_preview1:
            st.write("**1st Floor (Upper Level):**")
            first_slots = st.session_state.detection_system.floors.get("1st Floor", {})
            for slot_id, slot in first_slots.items():
                status_class = f"status-{slot.status.lower()}"
                st.markdown(f'<span class="{status_class}">🅿️ {slot_id}: {slot.status}</span>', 
                           unsafe_allow_html=True)
        
        with col_preview2:
            st.write("**Ground Floor (Lower Level):**")
            ground_slots = st.session_state.detection_system.floors.get("Ground Floor", {})
            for slot_id, slot in ground_slots.items():
                status_class = f"status-{slot.status.lower()}"
                st.markdown(f'<span class="{status_class}">🅿️ {slot_id}: {slot.status}</span>', 
                           unsafe_allow_html=True)

with col2:
    st.subheader("📊 Real-Time Statistics")
    
    # Display current stats
    if len(st.session_state.detection_system.floors) > 0:
        all_stats = st.session_state.detection_system.get_all_stats()
        
        # Overall building stats
        total_free = sum(stats['Free'] for stats in all_stats.values())
        total_occupied = sum(stats['Occupied'] for stats in all_stats.values())
        total_slots = sum(stats['Total'] for stats in all_stats.values())
        overall_occupancy = (total_occupied / total_slots * 100) if total_slots > 0 else 0
        
        st.metric("🏢 Building Overview", f"{total_slots} Total Slots")
        st.metric("🟢 Available", total_free)
        st.metric("🔴 Occupied", total_occupied)
        st.metric("📊 Overall Occupancy", f"{overall_occupancy:.1f}%")
        
        st.progress(overall_occupancy / 100)
        st.markdown("---")
        
        # Individual floor stats
        for floor_name in ["1st Floor", "Ground Floor"]:
            if floor_name in all_stats:
                stats = all_stats[floor_name]
                floor_display = f"{floor_name} ({'Upper' if '1st' in floor_name else 'Lower'})"
                st.markdown(f'<div class="floor-header">{floor_display}</div>', unsafe_allow_html=True)
                
                col_free, col_occupied = st.columns(2)
                with col_free:
                    st.metric("🟢 Free", stats['Free'])
                with col_occupied:
                    st.metric("🔴 Occupied", stats['Occupied'])
                
                if stats.get('Parking', 0) > 0 or stats.get('Leaving', 0) > 0:
                    col_trans1, col_trans2 = st.columns(2)
                    with col_trans1:
                        st.metric("🟡 Parking", stats['Parking'])
                    with col_trans2:
                        st.metric("🟠 Leaving", stats['Leaving'])
                
                occupancy_rate = stats.get('Occupancy_Rate', 0)
                st.metric("📈 Occupancy", f"{occupancy_rate:.1f}%")
                st.progress(occupancy_rate / 100)
                st.markdown("---")
    
    # System information
    st.subheader("ℹ️ System Status")
    
    if video_status["both_available"]:
        st.success("🎬 Integrated Videos: Available")
    elif video_status["ground_floor"] or video_status["first_floor"]:
        st.warning("⚠️ Partial Video Coverage")
    else:
        st.info("🎯 Simulation Mode Active")
    
    if st.session_state.model_loaded:
        st.success("🤖 AI Detection: Active")
    else:
        st.info("🎯 Simulation Detection: Active")
    
    st.metric("🏗️ Floors", len(st.session_state.detection_system.floors))
    st.metric("📍 Total Slots", sum(len(slots) for slots in st.session_state.detection_system.floors.values()))

# Analytics tabs
tab1, tab2, tab3 = st.tabs(["📋 Detailed View", "📈 Analytics", "📖 Setup Guide"])

with tab1:
    st.subheader("Detailed Slot Information")
    
    if len(st.session_state.detection_system.floors) > 0:
        selected_floor = st.selectbox("Select Floor", ["1st Floor", "Ground Floor"])
        
        if selected_floor in st.session_state.detection_system.floors:
            slots_data = []
            for slot_id, slot in st.session_state.detection_system.floors[selected_floor].items():
                transition_time = slot.get_remaining_transition_time()
                slots_data.append({
                    'Slot ID': slot.id,
                    'Current Status': slot.status,
                    'Confidence': f"{slot.confidence:.2f}",
                    'Transition Timer': f"{transition_time:.1f}s" if transition_time > 0 else "N/A",
                    'History Records': len(slot.status_history)
                })
            
            df = pd.DataFrame(slots_data)
            
            def highlight_status(val):
                if val == 'Free':
                    return 'background-color: #d4edda; color: #155724'
                elif val == 'Occupied':
                    return 'background-color: #f8d7da; color: #721c24'
                elif val in ['Parking', 'Leaving']:
                    return 'background-color: #fff3cd; color: #856404'
                return ''
            
            styled_df = df.style.applymap(highlight_status, subset=['Current Status'])
            st.dataframe(styled_df, use_container_width=True)

with tab2:
    st.subheader("Real-Time Analytics")
    
    if len(st.session_state.detection_system.floors) > 0:
        all_stats = st.session_state.detection_system.get_all_stats()
        
        # Multi-floor comparison
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["1st Floor (Upper)", "Ground Floor (Lower)"],
            specs=[[{"type": "pie"}, {"type": "pie"}]]
        )
        
        colors = ['#28a745', '#dc3545', '#ffc107', '#fd7e14']
        floor_order = ["1st Floor", "Ground Floor"]
        
        for i, floor_name in enumerate(floor_order):
            if floor_name in all_stats:
                stats = all_stats[floor_name]
                values = [stats['Free'], stats['Occupied'], stats['Parking'], stats['Leaving']]
                labels = ['Free', 'Occupied', 'Parking', 'Leaving']
                
                filtered_values = []
                filtered_labels = []
                filtered_colors = []
                
                for j, (val, label) in enumerate(zip(values, labels)):
                    if val > 0:
                        filtered_values.append(val)
                        filtered_labels.append(f"{label} ({val})")
                        filtered_colors.append(colors[j])
                
                if filtered_values:
                    fig.add_trace(
                        go.Pie(labels=filtered_labels, values=filtered_values,
                              name=floor_name, marker_colors=filtered_colors,
                              textinfo='label+percent'),
                        row=1, col=i+1
                    )
        
        fig.update_layout(title="Floor-wise Parking Distribution", height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("📖 Integration Setup Guide")
    
    st.markdown("""
    ## 🎯 How to Add Your Videos
    
    ### Step 1: Prepare Your Videos
    1. **Format**: Use MP4, AVI, or MOV files
    2. **Quality**: 720p or higher recommended
    3. **Duration**: Any length (system will loop automatically)
    4. **Content**: Ensure parking areas are clearly visible
    
    ### Step 2: File Placement
    
    **Option A: Create a 'videos' folder in your project directory**
    ```
    your_project/
    ├── parking_dashboard.py
    └── videos/
        ├── ground_floor_parking.mp4
        └── first_floor_parking.mp4
    ```
    
    **Option B: Modify the INTEGRATED_VIDEOS paths in the code**
    ```python
    INTEGRATED_VIDEOS = {
        "ground_floor": "/path/to/your/ground_floor_video.mp4",
        "first_floor": "/path/to/your/first_floor_video.mp4"
    }
    ```
    
    ### Step 3: Video Requirements
    - **Ground Floor Video**: Should show your ground level parking area
    - **1st Floor Video**: Should show your upper level parking area
    - **Naming**: Keep filenames consistent with the paths in INTEGRATED_VIDEOS
    
    ### Step 4: Slot Configuration (Optional)
    If your parking layout differs from the default:
    1. Use a tool to get polygon coordinates for each parking slot
    2. Update GROUND_FLOOR_SLOTS and FIRST_FLOOR_SLOTS arrays
    3. Each slot needs: `{"id": "unique_id", "polygon": [[x1,y1], [x2,y2], ...]}`
    
    ## 🔧 Current Configuration
    """)
    
    # Show current video paths
    st.code(f"""
Ground Floor Video: {INTEGRATED_VIDEOS['ground_floor']}
1st Floor Video: {INTEGRATED_VIDEOS['first_floor']}

Status:
- Ground Floor: {'✅ Found' if video_status['ground_floor'] else '❌ Not Found'}
- 1st Floor: {'✅ Found' if video_status['first_floor'] else '❌ Not Found'}
    """)
    
    st.markdown("""
    ## 🚀 Quick Test
    
    1. **Without Videos**: The system works in simulation mode
    2. **With One Video**: Uses real video + simulation for missing floor
    3. **With Both Videos**: Full integrated video analysis
    
    ## 📊 Features Available
    
    - **Real-time Detection**: YOLO-based vehicle detection
    - **Smart Transitions**: Prevents false status changes
    - **Multi-floor Support**: Proper floor ordering and visualization
    - **Live Statistics**: Real-time occupancy tracking
    - **Looping Playback**: Videos restart automatically
    - **Fallback Mode**: Simulation when videos unavailable
    
    ## 🔄 System States
    
    - **🎬 Integrated Video**: Using your real parking videos
    - **🎯 Simulation**: Generated parking scenarios
    - **🤖 AI Detection**: YOLO model active
    - **📊 Analytics**: Live statistics and trends
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <h4>🚗 Integrated Smart Parking Dashboard</h4>
    <p><b>Layout:</b> 1st Floor (Upper) → Ground Floor (Lower)</p>
    <p><b>Mode:</b> Integrated Videos with AI Detection</p>
    <p><b>Ready to Use:</b> Just click 'Start Detection' to begin!</p>
</div>
""", unsafe_allow_html=True)

# Auto-show system status on load
if 'status_shown' not in st.session_state:
    st.session_state.status_shown = True
    if video_status["both_available"]:
        st.success("🎉 System ready with integrated videos! Click 'Start Detection' to begin.")
    else:
        st.info("ℹ️ Running in simulation mode. Add videos to the specified paths for full functionality.")

#streamlit run parking_dashboard.py
