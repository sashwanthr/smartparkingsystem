import streamlit as st
import cv2
import numpy as np
import json
import threading
import queue
import time
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import os
from pathlib import Path
import tempfile
import io
from typing import Dict, List, Tuple, Optional
import logging

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
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .sidebar-status {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
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
        'ground_floor_slots': {},
        'first_floor_slots': {},
        'detection_history': [],
        'video_threads': {},
        'frame_queues': {},
        'stop_detection': False,
        'total_slots': 0,
        'free_slots': 0,
        'occupied_slots': 0,
        'transition_slots': 0,
        'last_update': datetime.now(),
        'config': {
            'confidence_threshold': 0.5,
            'transition_time': 10,
            'ground_floor_video': 'parking1.mp4',
            'first_floor_video': 'parking2.mp4',
            'ground_floor_slots': 'slots1.json',
            'first_floor_slots': 'slots2.json'
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

class MockYOLODetector:
    """Mock YOLO detector for demonstration purposes"""
    
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.vehicle_classes = [1, 2, 3, 5, 7]  # bike, car, motorcycle, bus, truck
        
    def detect(self, frame):
        """Mock detection that returns random detections"""
        height, width = frame.shape[:2]
        detections = []
        
        # Generate 2-5 random detections
        num_detections = np.random.randint(2, 6)
        
        for _ in range(num_detections):
            x1 = np.random.randint(0, width//2)
            y1 = np.random.randint(0, height//2)
            x2 = x1 + np.random.randint(50, 150)
            y2 = y1 + np.random.randint(30, 100)
            
            detection = {
                'bbox': [x1, y1, x2, y2],
                'confidence': np.random.uniform(0.3, 0.95),
                'class': np.random.choice(self.vehicle_classes)
            }
            detections.append(detection)
        
        return detections

class MultiFloorParkingDetector:
    """Main parking detection class"""
    
    def __init__(self):
        self.detector = MockYOLODetector()
        self.slots = {}
        self.detection_history = []
        self.transition_timers = {}
        
    def load_slots(self, slots_file: str, floor_name: str) -> bool:
        """Load parking slots from JSON file"""
        try:
            if os.path.exists(slots_file):
                with open(slots_file, 'r') as f:
                    slots_data = json.load(f)
                
                # Convert to internal format
                for slot_id, slot_info in slots_data.items():
                    self.slots[f"{floor_name}_{slot_id}"] = {
                        'id': slot_id,
                        'floor': floor_name,
                        'polygon': slot_info.get('coordinates', []),
                        'state': 'free',
                        'confidence': 0.0,
                        'last_detection': None,
                        'transition_start': None
                    }
                return True
            else:
                # Create sample slots if file doesn't exist
                self.create_sample_slots(floor_name)
                return False
        except Exception as e:
            st.error(f"Error loading slots from {slots_file}: {str(e)}")
            self.create_sample_slots(floor_name)
            return False
    
    def create_sample_slots(self, floor_name: str):
        """Create sample parking slots for demonstration"""
        sample_slots = {}
        
        # Create 10 sample slots with different positions
        for i in range(10):
            slot_id = f"slot_{i+1:02d}"
            # Create rectangular polygons at different positions
            x = (i % 5) * 120 + 50
            y = (i // 5) * 80 + 100
            
            sample_slots[f"{floor_name}_{slot_id}"] = {
                'id': slot_id,
                'floor': floor_name,
                'polygon': [[x, y], [x+100, y], [x+100, y+60], [x, y+60]],
                'state': 'free',
                'confidence': 0.0,
                'last_detection': None,
                'transition_start': None
            }
        
        self.slots.update(sample_slots)
    
    def point_in_polygon(self, point: Tuple[float, float], polygon: List[List[float]]) -> bool:
        """Check if a point is inside a polygon"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def calculate_overlap(self, bbox: List[float], polygon: List[List[float]]) -> float:
        """Calculate overlap between bounding box and polygon"""
        x1, y1, x2, y2 = bbox
        
        # Simple overlap calculation - check if bbox center is in polygon
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        if self.point_in_polygon((center_x, center_y), polygon):
            return 0.8  # High overlap if center is inside
        
        # Check if any corner of bbox is inside polygon
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        inside_count = sum(1 for corner in corners if self.point_in_polygon(corner, polygon))
        
        return inside_count / 4.0  # Return fraction of corners inside
    
    def update_slot_states(self, detections: List[Dict], floor_name: str):
        """Update slot states based on detections"""
        current_time = datetime.now()
        
        # Filter slots for current floor
        floor_slots = {k: v for k, v in self.slots.items() if v['floor'] == floor_name}
        
        for slot_key, slot in floor_slots.items():
            max_overlap = 0.0
            best_confidence = 0.0
            
            # Check all detections against this slot
            for detection in detections:
                if detection['confidence'] >= st.session_state.config['confidence_threshold']:
                    overlap = self.calculate_overlap(detection['bbox'], slot['polygon'])
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_confidence = detection['confidence']
            
            # Update slot state based on overlap
            previous_state = slot['state']
            
            if max_overlap > 0.3:  # Vehicle detected
                if previous_state == 'free':
                    slot['state'] = 'transition_to_occupied'
                    slot['transition_start'] = current_time
                elif previous_state == 'transition_to_occupied':
                    # Check if transition time has passed
                    if (current_time - slot['transition_start']).seconds >= st.session_state.config['transition_time']:
                        slot['state'] = 'occupied'
                elif previous_state == 'transition_to_free':
                    slot['state'] = 'transition_to_occupied'
                    slot['transition_start'] = current_time
                
                slot['confidence'] = best_confidence
                slot['last_detection'] = current_time
            else:  # No vehicle detected
                if previous_state == 'occupied':
                    slot['state'] = 'transition_to_free'
                    slot['transition_start'] = current_time
                elif previous_state == 'transition_to_free':
                    # Check if transition time has passed
                    if (current_time - slot['transition_start']).seconds >= st.session_state.config['transition_time']:
                        slot['state'] = 'free'
                elif previous_state == 'transition_to_occupied':
                    slot['state'] = 'transition_to_free'
                    slot['transition_start'] = current_time
                
                slot['confidence'] = 0.0
            
            # Update session state
            self.slots[slot_key] = slot
    
    def draw_slots_on_frame(self, frame: np.ndarray, floor_name: str) -> np.ndarray:
        """Draw parking slots on frame with color coding"""
        floor_slots = {k: v for k, v in self.slots.items() if v['floor'] == floor_name}
        
        for slot in floor_slots.values():
            polygon = np.array(slot['polygon'], np.int32)
            
            # Color based on state
            if slot['state'] == 'free':
                color = (0, 255, 0)  # Green
            elif slot['state'] == 'occupied':
                color = (0, 0, 255)  # Red
            else:  # Transition states
                color = (0, 165, 255)  # Orange
            
            # Draw polygon
            cv2.polylines(frame, [polygon], True, color, 2)
            cv2.fillPoly(frame, [polygon], (*color, 30))  # Semi-transparent fill
            
            # Draw slot ID
            center_x = int(np.mean([p[0] for p in polygon]))
            center_y = int(np.mean([p[1] for p in polygon]))
            cv2.putText(frame, slot['id'], (center_x-20, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def process_frame(self, frame: np.ndarray, floor_name: str) -> np.ndarray:
        """Process a single frame"""
        # Detect vehicles
        detections = self.detector.detect(frame)
        
        # Update slot states
        self.update_slot_states(detections, floor_name)
        
        # Draw slots
        processed_frame = self.draw_slots_on_frame(frame.copy(), floor_name)
        
        # Draw detection boxes
        for detection in detections:
            if detection['confidence'] >= st.session_state.config['confidence_threshold']:
                x1, y1, x2, y2 = detection['bbox']
                cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
                cv2.putText(processed_frame, f"{detection['confidence']:.2f}", 
                           (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Add statistics overlay
        self.add_statistics_overlay(processed_frame, floor_name)
        
        return processed_frame
    
    def add_statistics_overlay(self, frame: np.ndarray, floor_name: str):
        """Add statistics overlay to frame"""
        floor_slots = {k: v for k, v in self.slots.items() if v['floor'] == floor_name}
        
        total = len(floor_slots)
        free = sum(1 for slot in floor_slots.values() if slot['state'] == 'free')
        occupied = sum(1 for slot in floor_slots.values() if slot['state'] == 'occupied')
        transition = total - free - occupied
        
        # Draw background
        cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 120), (255, 255, 255), 2)
        
        # Draw text
        texts = [
            f"Floor: {floor_name.title()}",
            f"Total Slots: {total}",
            f"Free: {free}",
            f"Occupied: {occupied}",
            f"Transitioning: {transition}"
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(frame, text, (20, 30 + i*18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def download_from_google_drive(file_id: str, destination: str) -> bool:
    """Download file from Google Drive"""
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as file:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
                        status_text.text(f"Downloaded: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB")
        
        progress_bar.progress(1.0)
        status_text.text("Download completed!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return True
        
    except Exception as e:
        st.error(f"Error downloading file: {str(e)}")
        return False

def create_sample_slots_file(filename: str, floor_name: str):
    """Create a sample slots JSON file"""
    slots = {}
    
    # Create 10 sample slots
    for i in range(10):
        slot_id = f"slot_{i+1:02d}"
        x = (i % 5) * 120 + 50
        y = (i // 5) * 80 + 100
        
        slots[slot_id] = {
            "coordinates": [[x, y], [x+100, y], [x+100, y+60], [x, y+60]],
            "floor": floor_name,
            "type": "standard"
        }
    
    with open(filename, 'w') as f:
        json.dump(slots, f, indent=2)

def generate_mock_historical_data() -> pd.DataFrame:
    """Generate mock historical data for analytics"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H')
    
    data = []
    for date in dates:
        # Simulate occupancy patterns (higher during day, lower at night)
        hour = date.hour
        base_occupancy = 0.3 + 0.4 * np.sin((hour - 6) * np.pi / 12) if 6 <= hour <= 18 else 0.1
        
        ground_floor_occupied = max(0, min(10, int(10 * base_occupancy + np.random.normal(0, 2))))
        first_floor_occupied = max(0, min(10, int(10 * base_occupancy + np.random.normal(0, 2))))
        
        data.append({
            'timestamp': date,
            'ground_floor_total': 10,
            'ground_floor_occupied': ground_floor_occupied,
            'ground_floor_free': 10 - ground_floor_occupied,
            'first_floor_total': 10,
            'first_floor_occupied': first_floor_occupied,
            'first_floor_free': 10 - first_floor_occupied,
            'total_occupied': ground_floor_occupied + first_floor_occupied,
            'total_free': (10 - ground_floor_occupied) + (10 - first_floor_occupied)
        })
    
    return pd.DataFrame(data)

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
    
    # System status
    if st.session_state.detection_active:
        status_color = "🟢"
        status_text = "ACTIVE"
    else:
        status_color = "🔴"
        status_text = "INACTIVE"
    
    st.sidebar.markdown(f"""
    <div class="sidebar-status">
        <h4>{status_color} Detection Status: {status_text}</h4>
        <p><strong>Last Update:</strong> {st.session_state.last_update.strftime('%H:%M:%S')}</p>
        <p><strong>Total Slots:</strong> {st.session_state.total_slots}</p>
        <p><strong>Free Slots:</strong> {st.session_state.free_slots}</p>
        <p><strong>Occupied Slots:</strong> {st.session_state.occupied_slots}</p>
        <p><strong>Transitioning:</strong> {st.session_state.transition_slots}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Emergency stop
    if st.session_state.detection_active:
        if st.sidebar.button("🛑 Emergency Stop", type="secondary"):
            st.session_state.detection_active = False
            st.session_state.stop_detection = True
            st.rerun()

def configuration_page():
    """Configuration page"""
    st.header("⚙️ System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Video Files")
        
        # Ground floor video
        ground_video_path = st.text_input("Ground Floor Video Path", 
                                         value=st.session_state.config['ground_floor_video'])
        
        if st.button("📥 Download Ground Floor Video"):
            if download_from_google_drive("1yVBgp07Z8FfLzw5dd0SxW61zd1l55kjE", ground_video_path):
                st.success("Ground floor video downloaded successfully!")
            else:
                st.error("Failed to download ground floor video")
        
        # First floor video
        first_video_path = st.text_input("First Floor Video Path", 
                                        value=st.session_state.config['first_floor_video'])
        
        if st.button("📥 Download First Floor Video"):
            if download_from_google_drive("14njmPC4b81mOV02orKslziMRr2WL-qAM", first_video_path):
                st.success("First floor video downloaded successfully!")
            else:
                st.error("Failed to download first floor video")
        
        # Check file status
        st.subheader("File Status")
        files_to_check = [
            (ground_video_path, "Ground Floor Video"),
            (first_video_path, "First Floor Video"),
            (st.session_state.config['ground_floor_slots'], "Ground Floor Slots"),
            (st.session_state.config['first_floor_slots'], "First Floor Slots")
        ]
        
        for file_path, description in files_to_check:
            if os.path.exists(file_path):
                st.success(f"✅ {description}: Found")
            else:
                st.error(f"❌ {description}: Missing")
    
    with col2:
        st.subheader("Slot Configuration Files")
        
        ground_slots_path = st.text_input("Ground Floor Slots JSON", 
                                         value=st.session_state.config['ground_floor_slots'])
        
        if st.button("📝 Create Sample Ground Floor Slots"):
            create_sample_slots_file(ground_slots_path, "ground_floor")
            st.success("Sample ground floor slots file created!")
        
        first_slots_path = st.text_input("First Floor Slots JSON", 
                                        value=st.session_state.config['first_floor_slots'])
        
        if st.button("📝 Create Sample First Floor Slots"):
            create_sample_slots_file(first_slots_path, "first_floor")
            st.success("Sample first floor slots file created!")
        
        st.subheader("AI Model Settings")
        
        confidence_threshold = st.slider("Detection Confidence Threshold", 
                                        0.0, 1.0, 
                                        st.session_state.config['confidence_threshold'], 
                                        0.05)
        
        transition_time = st.number_input("Transition Time (seconds)", 
                                        min_value=1, max_value=60, 
                                        value=st.session_state.config['transition_time'])
        
        if st.button("💾 Save Configuration"):
            st.session_state.config.update({
                'ground_floor_video': ground_video_path,
                'first_floor_video': first_video_path,
                'ground_floor_slots': ground_slots_path,
                'first_floor_slots': first_slots_path,
                'confidence_threshold': confidence_threshold,
                'transition_time': transition_time
            })
            st.success("Configuration saved successfully!")
    
    st.subheader("System Control")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Detection", disabled=st.session_state.detection_active):
            # Initialize detector
            detector = MultiFloorParkingDetector()
            detector.load_slots(st.session_state.config['ground_floor_slots'], 'ground_floor')
            detector.load_slots(st.session_state.config['first_floor_slots'], 'first_floor')
            
            st.session_state.detection_active = True
            st.session_state.stop_detection = False
            st.success("Detection started!")
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Detection", disabled=not st.session_state.detection_active):
            st.session_state.detection_active = False
            st.session_state.stop_detection = True
            st.success("Detection stopped!")
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset System"):
            for key in ['detection_active', 'ground_floor_slots', 'first_floor_slots', 
                       'detection_history', 'video_threads', 'frame_queues']:
                if key in st.session_state:
                    if key == 'detection_active':
                        st.session_state[key] = False
                    elif key in ['ground_floor_slots', 'first_floor_slots', 'video_threads', 'frame_queues']:
                        st.session_state[key] = {}
                    elif key == 'detection_history':
                        st.session_state[key] = []
            st.success("System reset!")
            st.rerun()

def live_dashboard_page():
    """Live dashboard page"""
    st.header("📹 Live Dashboard")
    
    if not st.session_state.detection_active:
        st.warning("⚠️ Detection system is not active. Please start it from the Configuration page.")
        return
    
    # Statistics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{st.session_state.total_slots}</h3>
            <p>Total Slots</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%);">
            <h3>{st.session_state.free_slots}</h3>
            <p>Free Slots</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);">
            <h3>{st.session_state.occupied_slots}</h3>
            <p>Occupied Slots</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);">
            <h3>{st.session_state.transition_slots}</h3>
            <p>Transitioning</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Video displays
    st.subheader("Live Video Feeds")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Ground Floor")
        video_placeholder1 = st.empty()
        
        # Mock video display
        with video_placeholder1.container():
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            st.info("🎥 Ground Floor Live Feed\n\nVideo processing would display here in production mode.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### First Floor")
        video_placeholder2 = st.empty()
        
        # Mock video display
        with video_placeholder2.container():
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            st.info("🎥 First Floor Live Feed\n\nVideo processing would display here in production mode.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Auto-refresh
    if st.button("🔄 Refresh Status"):
        st.rerun()

def analytics_page():
    """Analytics page"""
    st.header("📊 Analytics Dashboard")
    
    # Generate mock data
    df = generate_mock_historical_data()
    
    # Time range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now().date())
    
    # Filter data
    df_filtered = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]
    
    # Overall statistics
    st.subheader("📈 Occupancy Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_occupancy = df_filtered['total_occupied'].mean()
        st.metric("Average Occupancy", f"{avg_occupancy:.1f} slots", f"{(avg_occupancy/20)*100:.1f}%")
    
    with col2:
        peak_occupancy = df_filtered['total_occupied'].max()
        st.metric("Peak Occupancy", f"{peak_occupancy} slots", f"{(peak_occupancy/20)*100:.1f}%")
    
    with col3:
        current_occupancy = df_filtered['total_occupied'].iloc[-1] if not df_filtered.empty else 0
        st.metric("Current Occupancy", f"{current_occupancy} slots", f"{(current_occupancy/20)*100:.1f}%")
    
    # Occupancy trend chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Occupancy Trend', 'Floor Comparison', 
                       'Occupancy Distribution', 'Hourly Patterns'),
        specs=[[{"secondary_y": True}, {"type": "bar"}],
               [{"type": "histogram"}, {"type": "heatmap"}]]
    )
    
    # Total occupancy trend
    fig.add_trace(
        go.Scatter(x=df_filtered['timestamp'], y=df_filtered['total_occupied'],
                  mode='lines', name='Occupied', line=dict(color='red')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_filtered['timestamp'], y=df_filtered['total_free'],
                  mode='lines', name='Free', line=dict(color='green')),
        row=1, col=1
    )
    
    # Floor comparison
    floor_comparison = pd.DataFrame({
        'Ground Floor': [df_filtered['ground_floor_occupied'].mean(), df_filtered['ground_floor_free'].mean()],
        'First Floor': [df_filtered['first_floor_occupied'].mean(), df_filtered['first_floor_free'].mean()]
    }, index=['Occupied', 'Free'])
    
    fig.add_trace(
        go.Bar(x=['Ground Floor', 'First Floor'], y=floor_comparison.loc['Occupied'],
               name='Avg Occupied', marker_color='red'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=['Ground Floor', 'First Floor'], y=floor_comparison.loc['Free'],
               name='Avg Free', marker_color='green'),
        row=1, col=2
    )
    
    # Occupancy distribution
    fig.add_trace(
        go.Histogram(x=df_filtered['total_occupied'], nbinsx=10,
                    name='Occupancy Distribution', marker_color='blue'),
        row=2, col=1
    )
    
    # Hourly patterns
    df_filtered['hour'] = df_filtered['timestamp'].dt.hour
    hourly_avg = df_filtered.groupby('hour')['total_occupied'].mean().reset_index()
    
    # Create heatmap data
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    hours = list(range(24))
    
    # Generate sample heatmap data
    heatmap_data = np.random.rand(7, 24) * 20
    
    fig.add_trace(
        go.Heatmap(z=heatmap_data, x=hours, y=days,
                  colorscale='RdYlGn_r', name='Hourly Heatmap'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="Parking Analytics Dashboard")
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed statistics table
    st.subheader("📋 Detailed Statistics")
    
    # Daily summary
    daily_summary = df_filtered.groupby(df_filtered['timestamp'].dt.date).agg({
        'total_occupied': ['mean', 'max', 'min'],
        'ground_floor_occupied': 'mean',
        'first_floor_occupied': 'mean'
    }).round(2)
    
    daily_summary.columns = ['Avg Occupied', 'Peak Occupied', 'Min Occupied', 
                           'Ground Floor Avg', 'First Floor Avg']
    
    st.dataframe(daily_summary, use_container_width=True)

def detailed_slots_page():
    """Detailed slots information page"""
    st.header("🅿️ Detailed Slot Information")
    
    # Floor filter
    floor_filter = st.selectbox("Select Floor", ["All", "Ground Floor", "First Floor"])
    
    # Mock slot data for display
    slots_data = []
    
    # Generate sample slot data
    for floor in ['ground_floor', 'first_floor']:
        for i in range(10):
            slot_id = f"slot_{i+1:02d}"
            state = np.random.choice(['free', 'occupied', 'transition_to_occupied', 'transition_to_free'], 
                                   p=[0.4, 0.4, 0.1, 0.1])
            confidence = np.random.uniform(0.0, 1.0) if state == 'occupied' else 0.0
            
            # Simulate transition timer
            transition_time = np.random.randint(0, 10) if 'transition' in state else 0
            
            slots_data.append({
                'Slot ID': slot_id,
                'Floor': floor.replace('_', ' ').title(),
                'Status': state.replace('_', ' ').title(),
                'Confidence': f"{confidence:.2f}",
                'Transition Timer': f"{transition_time}s" if transition_time > 0 else "-",
                'Last Detection': datetime.now() - timedelta(seconds=np.random.randint(0, 300))
            })
    
    df_slots = pd.DataFrame(slots_data)
    
    # Apply floor filter
    if floor_filter != "All":
        df_slots = df_slots[df_slots['Floor'] == floor_filter]
    
    # Status filter
    status_filter = st.multiselect(
        "Filter by Status",
        options=['Free', 'Occupied', 'Transition To Occupied', 'Transition To Free'],
        default=['Free', 'Occupied', 'Transition To Occupied', 'Transition To Free']
    )
    
    if status_filter:
        df_slots = df_slots[df_slots['Status'].isin(status_filter)]
    
    # Statistics summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_slots = len(df_slots)
        st.metric("Total Slots", total_slots)
    
    with col2:
        free_slots = len(df_slots[df_slots['Status'] == 'Free'])
        st.metric("Free Slots", free_slots, f"{(free_slots/total_slots*100):.1f}%" if total_slots > 0 else "0%")
    
    with col3:
        occupied_slots = len(df_slots[df_slots['Status'] == 'Occupied'])
        st.metric("Occupied Slots", occupied_slots, f"{(occupied_slots/total_slots*100):.1f}%" if total_slots > 0 else "0%")
    
    with col4:
        transition_slots = len(df_slots[df_slots['Status'].str.contains('Transition')])
        st.metric("Transitioning", transition_slots, f"{(transition_slots/total_slots*100):.1f}%" if total_slots > 0 else "0%")
    
    # Slots table with color coding
    st.subheader("Slot Details")
    
    def color_status(val):
        if val == 'Free':
            return 'background-color: #d4edda; color: #155724'
        elif val == 'Occupied':
            return 'background-color: #f8d7da; color: #721c24'
        elif 'Transition' in val:
            return 'background-color: #fff3cd; color: #856404'
        return ''
    
    styled_df = df_slots.style.applymap(color_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Export functionality
    if st.button("📊 Export Data"):
        csv = df_slots.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"parking_slots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def setup_guide_page():
    """Setup guide page"""
    st.header("📖 Setup Guide")
    
    st.markdown("""
    ## 🚀 Complete Installation Guide
    
    This guide will help you set up the Multi-Floor Parking Detection System for production use.
    """)
    
    # Installation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Local Setup", "☁️ Streamlit Cloud", "🐳 Docker", "🔧 Troubleshooting"])
    
    with tab1:
        st.subheader("Local Development Setup")
        
        st.markdown("""
        ### Prerequisites
        - Python 3.8 or higher
        - pip package manager
        - Git (optional, for cloning)
        
        ### Step 1: Install Dependencies
        ```bash
        pip install streamlit opencv-python ultralytics numpy pandas plotly requests
        ```
        
        ### Step 2: Download Required Files
        The system requires two main video files and slot configuration files:
        
        **Video Files:**
        - Ground Floor: Use the download button in Configuration page
        - First Floor: Use the download button in Configuration page
        
        **Slot Configuration Files:**
        - `slots1.json` - Ground floor parking slots
        - `slots2.json` - First floor parking slots
        
        ### Step 3: File Structure
        ```
        parking-detection/
        ├── app.py                 # Main Streamlit app
        ├── parking1.mp4          # Ground floor video
        ├── parking2.mp4          # First floor video
        ├── slots1.json           # Ground floor slots
        ├── slots2.json           # First floor slots
        └── requirements.txt      # Dependencies
        ```
        
        ### Step 4: Run the Application
        ```bash
        streamlit run app.py
        ```
        
        The application will be available at `http://localhost:8501`
        """)
    
    with tab2:
        st.subheader("Streamlit Cloud Deployment")
        
        st.markdown("""
        ### Prerequisites
        - GitHub account
        - Streamlit Cloud account (free)
        
        ### Step 1: Prepare Repository
        1. Create a GitHub repository
        2. Upload the application code (without large video files)
        3. Include `requirements.txt` with all dependencies
        
        ### Step 2: Requirements.txt
        ```txt
        streamlit==1.28.0
        opencv-python-headless==4.8.1.78
        ultralytics==8.0.196
        numpy==1.24.3
        pandas==2.0.3
        plotly==5.15.0
        requests==2.31.0
        ```
        
        ### Step 3: Deploy to Streamlit Cloud
        1. Go to [share.streamlit.io](https://share.streamlit.io)
        2. Connect your GitHub repository
        3. Select the main branch and app.py file
        4. Click "Deploy"
        
        ### Step 4: Configure Large Files
        Since video files are large, use the Google Drive download feature:
        - The app will automatically download videos when needed
        - No need to include large files in the repository
        
        ### Important Notes
        - Streamlit Cloud has resource limitations
        - For production use, consider upgrading to Streamlit Cloud Pro
        - Large video processing may require local deployment
        """)
    
    with tab3:
        st.subheader("Docker Deployment")
        
        st.markdown("""
        ### Dockerfile
        ```dockerfile
        FROM python:3.9-slim
        
        # Install system dependencies
        RUN apt-get update && apt-get install -y \\
            libglib2.0-0 \\
            libsm6 \\
            libxext6 \\
            libxrender-dev \\
            libgomp1 \\
            libglib2.0-0 \\
            libgtk-3-dev \\
            && rm -rf /var/lib/apt/lists/*
        
        # Set working directory
        WORKDIR /app
        
        # Copy requirements and install
        COPY requirements.txt .
        RUN pip install --no-cache-dir -r requirements.txt
        
        # Copy application code
        COPY . .
        
        # Expose port
        EXPOSE 8501
        
        # Run streamlit
        CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
        ```
        
        ### Build and Run
        ```bash
        # Build the image
        docker build -t parking-detection .
        
        # Run the container
        docker run -p 8501:8501 parking-detection
        ```
        
        ### Docker Compose (Optional)
        ```yaml
        version: '3.8'
        services:
          parking-detection:
            build: .
            ports:
              - "8501:8501"
            volumes:
              - ./data:/app/data
            environment:
              - STREAMLIT_SERVER_PORT=8501
        ```
        """)
    
    with tab4:
        st.subheader("Troubleshooting")
        
        st.markdown("""
        ### Common Issues and Solutions
        
        #### 🔴 Video Files Not Found
        **Problem:** Application shows "Video file not found" error
        
        **Solutions:**
        - Use the download buttons in Configuration page
        - Check file paths in configuration
        - Ensure files are in the correct directory
        
        #### 🔴 YOLO Model Loading Issues
        **Problem:** YOLO model fails to load or detect objects
        
        **Solutions:**
        - Check internet connection (model downloads automatically)
        - Verify ultralytics installation: `pip install ultralytics`
        - For offline use, download model manually
        
        #### 🔴 OpenCV Issues
        **Problem:** Video processing errors or display issues
        
        **Solutions:**
        - Install opencv-python-headless for cloud deployment
        - For local use: `pip install opencv-python`
        - Check video file format (MP4 recommended)
        
        #### 🔴 Memory Issues
        **Problem:** Application crashes or runs slowly
        
        **Solutions:**
        - Reduce video resolution
        - Adjust frame processing rate
        - Use smaller video files for testing
        - Consider upgrading hosting resources
        
        #### 🔴 Streamlit Cloud Limitations
        **Problem:** App exceeds resource limits
        
        **Solutions:**
        - Optimize code for efficiency
        - Reduce concurrent video processing
        - Use local deployment for heavy processing
        - Consider Streamlit Cloud Pro
        
        ### Performance Optimization Tips
        
        1. **Video Processing:**
           - Skip frames for faster processing
           - Use lower resolution videos
           - Implement frame caching
        
        2. **Detection Optimization:**
           - Adjust confidence threshold
           - Use region of interest (ROI)
           - Batch process detections
        
        3. **Memory Management:**
           - Clear unused variables
           - Use generators for large datasets
           - Implement proper garbage collection
        
        ### Getting Help
        
        If you encounter issues not covered here:
        - Check the GitHub repository for updates
        - Review Streamlit documentation
        - Check YOLO/Ultralytics documentation
        - Contact support team
        """)
    
    # System requirements
    st.subheader("📋 System Requirements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Minimum Requirements:**
        - CPU: 2 cores, 2.0 GHz
        - RAM: 4 GB
        - Storage: 2 GB free space
        - Python: 3.8+
        - Internet: Required for model download
        """)
    
    with col2:
        st.markdown("""
        **Recommended Requirements:**
        - CPU: 4+ cores, 3.0 GHz
        - RAM: 8+ GB
        - GPU: NVIDIA GPU (optional, for faster processing)
        - Storage: 10+ GB free space
        - Python: 3.9+
        - Internet: High-speed connection
        """)

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Show header
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
    
    # Update session state statistics (mock data for demo)
    if st.session_state.detection_active:
        # Simulate real-time updates
        st.session_state.total_slots = 20
        st.session_state.free_slots = np.random.randint(8, 15)
        st.session_state.occupied_slots = np.random.randint(5, 12)
        st.session_state.transition_slots = st.session_state.total_slots - st.session_state.free_slots - st.session_state.occupied_slots
        st.session_state.last_update = datetime.now()
    
    # Display selected page
    pages[selected_page]()
    
    # Auto-refresh for live pages
    if selected_page == "📹 Live Dashboard" and st.session_state.detection_active:
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main()
