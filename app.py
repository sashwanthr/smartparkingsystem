# Reset to beginning
            self.cap_ground.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.cap_first.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Get FPS from video
            fps = int(self.cap_ground.get(cv2.CAP_PROP_FPS)) or 30
            self.detection_params['fps'] = fps
            self.transition_frames = self.detection_params['transition_time_seconds'] * fps
            
            self.is_running = True
            self.frame_count = 0
            
            print(f"Detection started successfully at {fps} FPS")
            return True
            
        except Exception as e:
            print(f"Error starting detection: {e}")
            if hasattr(self, 'cap_ground') and self.cap_ground:
                self.cap_ground.release()
            if hasattr(self, 'cap_first') and self.cap_first:
                self.cap_first.release()
            return False
    
    def stop_detection(self):
        """Stop video detection"""
        self.is_running = False
        if self.cap_ground:
            self.cap_ground.release()
        if self.cap_first:
            self.cap_first.release()
        print("Detection stopped")
    
    def get_next_frames(self):
        """Get next frames from both videos"""
        if not self.is_running or not self.cap_ground or not self.cap_first:
            return False, None, None
            
        ret1, ground_frame = self.cap_ground.read()
        ret2, first_floor_frame = self.cap_first.read()
        
        if not ret1 or not ret2:
            return False, None, None
            
        return True, ground_frame, first_floor_frame
    
    def get_parking_statistics(self):
        """Get current parking statistics"""
        stats = {
            'ground_floor': {
                'total': len(self.ground_floor_slots),
                'free': sum(1 for slot in self.ground_floor_slots if slot['state'] == 'free'),
                'occupied': sum(1 for slot in self.ground_floor_slots if slot['state'] == 'occupied'),
                'transitioning': sum(1 for slot in self.ground_floor_slots if slot['state'] in ['transition_to_occupied', 'transition_to_free'])
            },
            'first_floor': {
                'total': len(self.first_floor_slots),
                'free': sum(1 for slot in self.first_floor_slots if slot['state'] == 'free'),
                'occupied': sum(1 for slot in self.first_floor_slots if slot['state'] == 'occupied'),
                'transitioning': sum(1 for slot in self.first_floor_slots if slot['state'] in ['transition_to_occupied', 'transition_to_free'])
            }
        }
        
        stats['total'] = {
            'total': stats['ground_floor']['total'] + stats['first_floor']['total'],
            'free': stats['ground_floor']['free'] + stats['first_floor']['free'],
            'occupied': stats['ground_floor']['occupied'] + stats['first_floor']['occupied'],
            'transitioning': stats['ground_floor']['transitioning'] + stats['first_floor']['transitioning']
        }
        
        return stats

# --- Page Functions ---

def download_from_google_drive(file_id, destination):
    """Download large files from Google Drive with proper handling"""
    try:
        import urllib.request
        import urllib.parse
        import re
        
        # For large files, Google Drive requires a confirmation token
        session_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Initiating download...")
        
        # First request to get the confirmation token for large files
        response = urllib.request.urlopen(session_url)
        content = response.read().decode('utf-8')
        
        # Look for the confirmation token
        confirm_token = None
        if 'confirm=' in content:
            # Extract confirmation token using regex
            token_match = re.search(r'confirm=([a-zA-Z0-9_-]+)', content)
            if token_match:
                confirm_token = token_match.group(1)
        
        # Download URL with confirmation token if needed
        if confirm_token:
            download_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
            status_text.text("Large file detected, processing...")
        else:
            download_url = session_url
        
        # Download with progress tracking
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                progress = min(1.0, (block_num * block_size) / total_size)
                progress_bar.progress(progress)
                downloaded_mb = (block_num * block_size) / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                status_text.text(f"Downloading... {progress*100:.1f}% ({downloaded_mb:.1f}MB / {total_mb:.1f}MB)")
            else:
                # If total size is unknown, show downloaded amount
                downloaded_mb = (block_num * block_size) / (1024 * 1024)
                status_text.text(f"Downloading... {downloaded_mb:.1f}MB downloaded")
        
        urllib.request.urlretrieve(download_url, destination, progress_hook)
        
        # Verify file was downloaded properly
        if os.path.exists(destination) and os.path.getsize(destination) > 1024:  # At least 1KB
            file_size_mb = os.path.getsize(destination) / (1024 * 1024)
            progress_bar.progress(1.0)
            status_text.text(f"Download complete! File size: {file_size_mb:.1f}MB")
            return True
        else:
            st.error("Download failed: File is too small or corrupted")
            return False
            
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
        return False

def show_config_page():
    st.markdown("""
    <h1 class="main-header">
        Smart Parking Management System
    </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("Configure your multi-floor parking detection system with real video files and slot layouts.")

    st.subheader("Video Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Ground Floor Video**")
        ground_video = st.text_input("Ground Floor Video Path", value="parking1.mp4", key="ground_video")
        
        # Video file status check with download option
        if os.path.exists(ground_video):
            file_size = os.path.getsize(ground_video) / (1024*1024)  # MB
            if file_size > 1.0:  # Valid file size
                st.success(f"Found: {ground_video}")
                st.info(f"File size: {file_size:.1f} MB")
            else:
                st.error(f"File corrupted (too small): {ground_video}")
                if st.button("Re-download Ground Floor Video", key="redownload_ground"):
                    if os.path.exists(ground_video):
                        os.remove(ground_video)  # Remove corrupted file
                    if download_from_google_drive('1yVBgp07Z8FfLzw5dd0SxW61zd1l55kjE', 'parking1.mp4'):
                        st.success("Download completed!")
                        st.rerun()
        else:
            st.error(f"File not found: {ground_video}")
            if st.button("Download Ground Floor Video", key="download_ground"):
                if download_from_google_drive('1yVBgp07Z8FfLzw5dd0SxW61zd1l55kjE', 'parking1.mp4'):
                    st.success("Download completed!")
                    st.rerun()
        
    with col2:
        st.write("**First Floor Video**")
        first_video = st.text_input("First Floor Video Path", value="parking2.mp4", key="first_video")
        
        if os.path.exists(first_video):
            file_size = os.path.getsize(first_video) / (1024*1024)  # MB
            if file_size > 1.0:  # Valid file size
                st.success(f"Found: {first_video}")
                st.info(f"File size: {file_size:.1f} MB")
            else:
                st.error(f"File corrupted (too small): {first_video}")
                if st.button("Re-download First Floor Video", key="redownload_first"):
                    if os.path.exists(first_video):
                        os.remove(first_video)  # Remove corrupted file
                    if download_from_google_drive('14njmPC4b81mOV02orKslziMRr2WL-qAM', 'parking2.mp4'):
                        st.success("Download completed!")
                        st.rerun()
        else:
            st.error(f"File not found: {first_video}")
            if st.button("Download First Floor Video", key="download_first"):
                if download_from_google_drive('14njmPC4b81mOV02orKslziMRr2WL-qAM', 'parking2.mp4'):
                    st.success("Download completed!")
                    st.rerun()
                    
    # Download all videos at once
    col_download1, col_download2, col_download3 = st.columns(3)
    with col_download2:
        if st.button("Download All Videos", use_container_width=True):
            with st.spinner("Downloading videos..."):
                success = True
                
                # Download ground floor video if needed
                if not os.path.exists('parking1.mp4') or os.path.getsize('parking1.mp4') <= 1024:
                    st.info("Downloading ground floor video...")
                    if os.path.exists('parking1.mp4'):
                        os.remove('parking1.mp4')  # Remove corrupted file
                    success &= download_from_google_drive('1yVBgp07Z8FfLzw5dd0SxW61zd1l55kjE', 'parking1.mp4')
                
                # Download first floor video if needed  
                if not os.path.exists('parking2.mp4') or os.path.getsize('parking2.mp4') <= 1024:
                    st.info("Downloading first floor video...")
                    if os.path.exists('parking2.mp4'):
                        os.remove('parking2.mp4')  # Remove corrupted file
                    success &= download_from_google_drive('14njmPC4b81mOV02orKslziMRr2WL-qAM', 'parking2.mp4')
                    
                if success:
                    st.success("All videos downloaded successfully!")
                    st.rerun()
                else:
                    st.error("Some downloads failed. Please try downloading individually.")

    st.subheader("Slot Configuration")
    
    col3, col4 = st.columns(2)
    with col3:
        st.write("**Ground Floor Slots**")
        ground_slots = st.text_input("Ground Floor Slots JSON", value="slots1.json", key="ground_slots")
        if os.path.exists(ground_slots):
            st.success(f"Found: {ground_slots}")
            # Show slot count
            try:
                with open(ground_slots, 'r') as f:
                    slots_data = json.load(f)
                st.info(f"Loaded {len(slots_data)} parking slots")
            except:
                st.warning("Could not read slot data")
        else:
            st.error(f"File not found: {ground_slots}")
            
    with col4:
        st.write("**First Floor Slots**")
        first_slots = st.text_input("First Floor Slots JSON", value="slots2.json", key="first_slots")
        if os.path.exists(first_slots):
            st.success(f"Found: {first_slots}")
            # Show slot count
            try:
                with open(first_slots, 'r') as f:
                    slots_data = json.load(f)
                st.info(f"Loaded {len(slots_data)} parking slots")
            except:
                st.warning("Could not read slot data")
        else:
            st.error(f"File not found: {first_slots}")

    st.subheader("AI Model Settings")
    
    col5, col6 = st.columns(2)
    with col5:
        model_path = st.text_input("YOLO Model Path", value="yolov8n.pt", key="model_path")
        if os.path.exists(model_path):
            st.success(f"Found: {model_path}")
        else:
            st.error(f"Model not found: {model_path}")
        
    with col6:
        confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05, key="confidence")
        transition_time = st.slider("Transition Time (seconds)", 5, 30, 10, 1, key="transition_time")
        
    st.subheader("System Control")
    
    # Check if all files exist and are valid
    ground_exists = os.path.exists(ground_video) and os.path.getsize(ground_video) > 1024
    first_exists = os.path.exists(first_video) and os.path.getsize(first_video) > 1024
    slots1_exists = os.path.exists(ground_slots)
    slots2_exists = os.path.exists(first_slots)
    model_exists = os.path.exists(model_path)
    
    files_exist = all([ground_exists, first_exists, slots1_exists, slots2_exists, model_exists])
    
    # Display file status summary
    if not files_exist:
        st.error("Missing or corrupted files detected:")
        if not ground_exists:
            st.error(f"❌ Ground floor video: {ground_video}")
        if not first_exists:
            st.error(f"❌ First floor video: {first_video}")
        if not slots1_exists:
            st.error(f"❌ Ground floor slots: {ground_slots}")
        if not slots2_exists:
            st.error(f"❌ First floor slots: {first_slots}")
        if not model_exists:
            st.error(f"❌ YOLO model: {model_path}")
    else:
        st.success("All required files are ready!")
    
    col7, col8, col9 = st.columns(3)
    with col7:
        start_button = st.button("Start Detection", use_container_width=True, disabled=not files_exist)
        if start_button:
            if files_exist:
                try:
                    with st.spinner("Initializing detection system..."):
                        # Initialize detector with user settings
                        st.session_state.detector = MultiFloorParkingDetector(
                            model_path=model_path,
                            ground_slots_json=ground_slots,
                            first_floor_slots_json=first_slots,
                            ground_video=ground_video,
                            first_floor_video=first_video,
                            confidence_threshold=confidence
                        )
                        
                        # Update transition time
                        st.session_state.detector.detection_params['transition_time_seconds'] = transition_time
                        
                        if st.session_state.detector.start_detection():
                            st.session_state.is_running = True
                            st.session_state.page = 'Live Dashboard'
                            st.success("Detection started successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to start detection! Check video files.")
                            
                except Exception as e:
                    st.error(f"Error initializing detector: {str(e)}")
                    st.error("Please check that all files are valid and accessible.")
            else:
                st.error("Cannot start detection - missing required files!")
    
    with col8:
        if st.button("Stop Detection", use_container_width=True):
            if st.session_state.detector:
                st.session_state.detector.stop_detection()
            st.session_state.is_running = False
            st.session_state.page = 'Configuration'
            st.success("Detection stopped")
            st.rerun()
    
    with col9:
        if st.button("Reset System", use_container_width=True):
            if st.session_state.detector:
                st.session_state.detector.stop_detection()
            st.session_state.is_running = False
            st.session_state.detector = None
            st.session_state.page = 'Configuration'
            st.success("System reset")
            st.rerun()
    
    if not files_exist:
        st.warning("Please ensure all required files are in your project directory before starting detection.")
        
        st.markdown("""
        ### Required File Structure:
        ```
        your_project_folder/
        ├── app.py (this file)
        ├── parking1.mp4 (ground floor video)
        ├── parking2.mp4 (first floor video)
        ├── slots1.json (ground floor parking slots)
        ├── slots2.json (first floor parking slots)
        └── yolov8n.pt (YOLO model weights)
        ```
        """)

def show_live_dashboard():
    st.markdown("""
    <h1 class="main-header">
        Live Multi-Floor Detection
    </h1>
    """, unsafe_allow_html=True)
    
    if not st.session_state.detector or not st.session_state.is_running:
        st.error("Detection system not running. Please go to Configuration page and start detection.")
        return
    
    # Control buttons
    col_control1, col_control2, col_control3 = st.columns([1, 1, 2])
    with col_control1:
        if st.button("Pause", use_container_width=True):
            st.session_state.is_running = False
    with col_control2:
        if st.button("Stop", use_container_width=True):
            st.session_state.detector.stop_detection()
            st.session_state.is_running = False
            st.session_state.page = 'Configuration'
            st.rerun()
    
    # Statistics display
    if st.session_state.detector:
        stats = st.session_state.detector.get_parking_statistics()
        
        # Overall statistics
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("Total Spots", stats['total']['total'])
        with col_stat2:
            st.metric("Free Spots", stats['total']['free'], delta=None)
        with col_stat3:
            st.metric("Occupied Spots", stats['total']['occupied'], delta=None)
        with col_stat4:
            st.metric("In Transition", stats['total']['transitioning'], delta=None)
    
    # Video display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("First Floor")
        first_floor_placeholder = st.empty()
        
        if st.session_state.detector:
            st.markdown(f"**Total:** {stats['first_floor']['total']} | **Free:** {stats['first_floor']['free']} | **Occupied:** {stats['first_floor']['occupied']}")
    
    with col2:
        st.subheader("Ground Floor") 
        ground_floor_placeholder = st.empty()
        
        if st.session_state.detector:
            st.markdown(f"**Total:** {stats['ground_floor']['total']} | **Free:** {stats['ground_floor']['free']} | **Occupied:** {stats['ground_floor']['occupied']}")
    
    # Detection loop
    if st.session_state.is_running and st.session_state.detector:
        # Process frames
        try:
            ret, ground_frame, first_floor_frame = st.session_state.detector.get_next_frames()
            
            if ret:
                # Process both frames
                ground_result, first_floor_result = st.session_state.detector.process_frame_dual(
                    ground_frame, first_floor_frame
                )
                
                # Display frames
                with col1:
                    first_floor_placeholder.image(first_floor_result, channels="BGR", use_column_width=True)
                with col2:
                    ground_floor_placeholder.image(ground_result, channels="BGR", use_column_width=True)
                
                # Auto-refresh for continuous detection
                time.sleep(0.033)  # ~30 FPS
                st.rerun()
            else:
                st.warning("End of video reached or error reading frames.")
                st.session_state.is_running = False
                
        except Exception as e:
            st.error(f"Detection error: {str(e)}")
            st.session_state.is_running = False

def show_detailed_slots():
    st.markdown("""
    <h1 class="main-header">
        Detailed Slot Information
    </h1>
    """, unsafe_allow_html=True)
    
    if not st.session_state.detector:
        st.warning("Please start detection first to view slot details.")
        return
    
    # Floor selection
    floor_filter = st.selectbox("Select Floor", ["All Floors", "Ground Floor", "1st Floor"])
    
    # Get all slots data
    all_spots = st.session_state.detector.all_slots
    
    # Filter by floor if needed
    if floor_filter != "All Floors":
        all_spots = [spot for spot in all_spots if spot['floor'] == floor_filter.replace(" Floor", "")]
    
    # Create dataframe
    spot_data = []
    for spot in all_spots:
        # Determine status display
        status = spot['state']
        if status in ['transition_to_occupied', 'transition_to_free']:
            frames_passed = st.session_state.detector.frame_count - spot.get('transition_start_frame', 0)
            seconds_remaining = max(0, (st.session_state.detector.transition_frames - frames_passed) // st.session_state.detector.detection_params['fps'])
            status_display = f"{status.replace('_', ' ').title()} ({seconds_remaining}s)"
        else:
            status_display = status.title()
        
        spot_data.append({
            "Slot ID": spot['id'],
            "Floor": spot['floor'],
            "Status": status_display,
            "Confidence": f"{spot['confidence']:.2f}",
            "Stable": "Yes" if spot['stable'] else "No",
            "Area": spot['area'],
            "Center": f"({spot['center'][0]}, {spot['center'][1]})"
        })
    
    df = pd.DataFrame(spot_data)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        free_count = len([s for s in all_spots if s['state'] == 'free'])
        st.metric("Free Slots", free_count)
    with col2:
        occupied_count = len([s for s in all_spots if s['state'] == 'occupied'])
        st.metric("Occupied Slots", occupied_count)
    with col3:
        transition_count = len([s for s in all_spots if s['state'] in ['transition_to_occupied', 'transition_to_free']])
        st.metric("In Transition", transition_count)
    with col4:
        total_count = len(all_spots)
        st.metric("Total Slots", total_count)
    
    # Display table
    st.dataframe(df, use_container_width=True)
    
    # Refresh button
    if st.button("Refresh Data"):
        st.rerun()

def generate_historical_data(detector=None):
    """Generate historical data for analytics"""
    dates = [datetime.now() - timedelta(hours=i) for i in range(24)]
    data = []
    
    if detector:
        total_spots = len(detector.all_slots)
    else:
        total_spots = 16  # Default for demo
    
    for dt in reversed(dates):
        hour = dt.hour
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            occupied = random.randint(int(total_spots * 0.7), total_spots)
        elif 22 <= hour or hour <= 6:  # Night hours
            occupied = random.randint(0, int(total_spots * 0.3))
        else:
            occupied = random.randint(int(total_spots * 0.2), int(total_spots * 0.6))
        
        data.append({
            'Timestamp': dt,
            'Occupied Spots': occupied,
            'Free Spots': total_spots - occupied,
            'Occupancy Rate (%)': (occupied / total_spots) * 100,
            'Total Spots': total_spots
        })
    return pd.DataFrame(data)

def show_analytics_page():
    st.markdown("""
    <h1 class="main-header">
        Parking Analytics
    </h1>
    """, unsafe_allow_html=True)
    
    # Generate historical data
    historical_df = generate_historical_data(st.session_state.detector)
    
    # Current statistics if detector is running
    if st.session_state.detector and st.session_state.is_running:
        current_stats = st.session_state.detector.get_parking_statistics()
        
        st.subheader("Current Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Ground Floor**")
            st.metric("Free", current_stats['ground_floor']['free'])
            st.metric("Occupied", current_stats['ground_floor']['occupied'])
            st.metric("Transitioning", current_stats['ground_floor']['transitioning'])
            
        with col2:
            st.markdown("**First Floor**")
            st.metric("Free", current_stats['first_floor']['free'])
            st.metric("Occupied", current_stats['first_floor']['occupied'])
            st.metric("Transitioning", current_stats['first_floor']['transitioning'])
            
        with col3:
            st.markdown("**Total**")
            occupancy_rate = (current_stats['total']['occupied'] / current_stats['total']['total']) * 100
            st.metric("Occupancy Rate", f"{occupancy_rate:.1f}%")
            st.metric("Available", current_stats['total']['free'])
    
    # Historical charts
    st.subheader("Historical Trends")
    
    # Occupancy rate over time
    fig1 = px.area(
        historical_df,
        x='Timestamp',
        y='Occupancy Rate (%)',
        title='Parking Lot Occupancy Over the Last 24 Hours',
        color_discrete_sequence=['#667eea']
    )
    fig1.update_layout(
        xaxis_title="Time",
        yaxis_title="Occupancy Rate (%)",
        hovermode="x unified"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Occupied vs Free spots
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=historical_df['Timestamp'],
        y=historical_df['Occupied Spots'],
        mode='lines+markers',
        name='Occupied',
        line=dict(color='red', width=2)
    ))
    fig2.add_trace(go.Scatter(
        x=historical_df['Timestamp'],
        y=historical_df['Free Spots'],
        mode='lines+markers',
        name='Free',
        line=dict(color='green', width=2)
    ))
    fig2.update_layout(
        title='Occupied vs Free Spots Over Time',
        xaxis_title='Time',
        yaxis_title='Number of Spots',
        hovermode='x unified'
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Data table
    st.subheader("Historical Data")
    st.dataframe(historical_df.set_index('Timestamp').tail(12), use_container_width=True)

def show_setup_guide():
    st.markdown("""
    <h1 class="main-header">
        Setup Guide
    </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## Complete Setup Guide for Multi-Floor Parking Detection
    
    ### File Structure (GitHub Repository)
    Since GitHub has a 25MB file limit, store only these files in your repository:
    
    ```
    /your-github-repo/
    ├── app.py                    # This Streamlit dashboard
    ├── slots1.json              # Ground floor parking slots
    ├── slots2.json              # First floor parking slots
    ├── requirements.txt         # Python dependencies
    ├── packages.txt             # System dependencies
    └── README.md                # Setup instructions
    ```
    
    ### Video Files (External Storage)
    Your video files are hosted on Google Drive and will download automatically.
    
    ### Installation Steps
    
    **1. Create requirements.txt:**
    ```txt
    streamlit>=1.28.0
    ultralytics>=8.0.0
    opencv-python-headless>=4.8.0
    pandas>=2.0.0
    plotly>=5.15.0
    numpy>=1.24.0
    Pillow>=9.0.0
    torch>=2.0.0
    torchvision>=0.15.0
    ```
    
    **2. Create packages.txt:**
    ```txt
    libgl1-mesa-glx
    libglib2.0-0
    libsm6
    libxext6
    libxrender-dev
    libgomp1
    ```
    
    **3. Deploy to Streamlit Cloud:**
    1. Push code to GitHub (without video files)
    2. Connect repository to Streamlit Cloud
    3. Deploy directly from GitHub
    4. Videos download automatically on first use
    
    **4. Local Development:**
    ```bash
    git clone your-repo-url
    cd your-repo
    pip install -r requirements.txt
    streamlit run app.py
    ```
    """)
    
    # File checker section
    st.subheader("File Status Checker")
    
    required_files = [
        "parking1.mp4",
        "parking2.mp4", 
        "slots1.json",
        "slots2.json",
        "yolov8n.pt"
    ]
    
    col1, col2 = st.columns(2)
    
    for i, file_name in enumerate(required_files):
        with col1 if i % 2 == 0 else col2:
            if os.path.exists(file_name):
                st.success(f"Found: {file_name}")
            else:
                st.error(f"Missing: {file_name}")

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("## Navigation")
    
    if st.session_state.is_running:
        page_options = ["Live Dashboard", "Analytics", "Detailed Slots", "Setup Guide", "Configuration"]
    else:
        page_options = ["Configuration", "Setup Guide"]
    
    selected_page = st.radio("Go to:", page_options)
    st.session_state.page = selected_page
    
    # System status
    st.markdown("---")
    st.markdown("## System Status")
    if st.session_state.is_running:
        st.success("Detection Running")
        if st.session_state.detector:
            stats = st.session_state.detector.get_parking_statistics()
            st.metric("Frame", st.session_state.detector.frame_count)
            st.metric("Total Occupancy", f"{(stats['total']['occupied']/stats['total']['total']*100):.1f}%")
    else:
        st.error("Detection Stopped")
    
    # Emergency stop
    st.markdown("---")
    if st.button("Emergency Stop", use_container_width=True, key="emergency_stop"):
        if st.session_state.detector:
            st.session_state.detector.stop_detection()
        st.session_state.is_running = False
        st.session_state.page = 'Configuration'
        st.rerun()

# --- Main Page Router ---
if st.session_state.page == "Configuration":
    show_config_page()
elif st.session_state.page == "Live Dashboard":
    show_live_dashboard()
elif st.session_state.page == "Analytics":
    show_analytics_page()
elif st.session_state.page == "Detailed Slots":
    show_detailed_slots()
elif st.session_state.page == "Setup Guide":
    show_setup_guide()

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8rem;'>
    Smart Parking Management System | Multi-Floor Detection with AI
</div>
""", unsafe_allow_html=True)import streamlit as st
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
        """
        Initialize the multi-floor parking detector with transition states
        """
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
