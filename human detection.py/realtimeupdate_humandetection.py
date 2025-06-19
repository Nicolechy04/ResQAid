import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import threading
from queue import Queue

class PersonTracker:
    def __init__(self, camera_index=0, model_path="yolov8n.pt"):
        print("🚀 Initializing Person Tracker...")
        
        # Load YOLOv8 model
        print("📦 Loading YOLOv8 model...")
        self.model = YOLO(model_path)
        print("✅ Model loaded successfully!")
        
        # Initialize webcam with optimized settings
        print(f"📹 Connecting to camera {camera_index}...")
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise Exception(f"❌ Could not open camera {camera_index}")
        
        # Optimize camera settings for performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        # Additional performance optimizations
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Camera initialized: {self.width}x{self.height} at {self.fps:.1f} FPS")
        
        # Tracking setup
        self.track_history = {}
        self.frame_count = 0
        self.start_time = time.time()
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define grid coordinates
        self.grid_coordinates = {
            'G11': {'latitude': 3.111731, 'longitude': 101.644608},
            'G12': {'latitude': 3.111731, 'longitude': 101.65363},
            'G13': {'latitude': 3.111731, 'longitude': 101.662652},
            'G21': {'latitude': 3.12074, 'longitude': 101.644608},
            'G22': {'latitude': 3.12074, 'longitude': 101.65363},  
            'G23': {'latitude': 3.12074, 'longitude': 101.662652},
            'G31': {'latitude': 3.129749, 'longitude': 101.644608},
            'G32': {'latitude': 3.129749, 'longitude': 101.65363},
            'G33': {'latitude': 3.129749, 'longitude': 101.662652}
        }
        
        # Initialize Google Sheets in background thread
        self.sheets_ready = False
        self.sheets_thread = threading.Thread(target=self.init_google_sheets, daemon=True)
        self.sheets_thread.start()
        
        # Performance optimization settings
        self.detection_interval = 2  # Run detection every 2 frames
        self.detection_counter = 0
        self.last_detected_people = 0
        
        # Batch updates for Google Sheets
        self.sheet_update_queue = Queue()
        self.sheet_update_thread = None
        
        # UI update optimization
        self.ui_update_counter = 0
        self.ui_update_interval = 10  # Update UI every 10 frames
        
        # Pre-calculated values
        self.stats_overlay = None
        self.instructions_overlay = None
        


    def init_google_sheets(self):
        # Initialize Google Sheets connection in background
        try:
            print("🔗 Connecting to Google Sheets...")
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
            client = gspread.authorize(creds)
            self.sheet = client.open("Route_Planning").sheet1
            
            # Get header row and find column indices
            header = self.sheet.row_values(1)
            print(f"📋 Sheet headers: {header}")
            
            # Find column indices
            try:
                self.grid_id_col = header.index("grid_id") + 1
                self.latitude_col = header.index("latitude") + 1
                self.longitude_col = header.index("longitude") + 1
                self.human_detection_col = header.index("human_detection") + 1
                self.timestamp_col = header.index("timestamp1") + 1
                self.s_latitude1_col = header.index("s_latitude1") + 1
                self.s_longtitude1_col = header.index("s_longtitude1") + 1
                
                print(f"✅ Column mapping completed")
                
            except ValueError as e:
                raise Exception(f"❌ Required column not found in sheet header: {e}")
            
            # Initialize/populate the sheet with grid data if needed
            self.setup_grid_data()
            
            # Start sheet update thread
            self.sheet_update_thread = threading.Thread(target=self.sheet_update_worker, daemon=True)
            self.sheet_update_thread.start()
            
            self.sheets_ready = True
            print("✅ Google Sheets ready")
            
        except Exception as e:
            print(f"⚠️ Google Sheets initialization failed: {e}")
            self.sheets_ready = False
    
    def setup_grid_data(self):
        """Initialize the sheet with grid data if not already present"""
        print("🔧 Setting up grid data in sheet...")
        
        try:
            # Get all current grid_ids in the sheet
            grid_id_values = self.sheet.col_values(self.grid_id_col)
            existing_grids = set(grid_id_values[1:])
            
            # Add missing grids
            for grid_id, coords in self.grid_coordinates.items():
                if grid_id not in existing_grids:
                    next_row = len(grid_id_values) + 1
                    
                    updates = [
                        (next_row, self.grid_id_col, grid_id),
                        (next_row, self.latitude_col, coords['latitude']),
                        (next_row, self.longitude_col, coords['longitude']),
                        (next_row, self.human_detection_col, 0),
                        (next_row, self.timestamp_col, "")
                    ]
                    
                    for row, col, value in updates:
                        self.sheet.update_cell(row, col, value)
                    
                    print(f"   ➕ Added {grid_id} to row {next_row}")
            
            self.create_row_mapping()
            
        except Exception as e:
            print(f"⚠️ Error setting up grid data: {e}")
    
    def create_row_mapping(self):
        """Create mapping from grid_id to row number for efficient updates"""
        try:
            grid_id_values = self.sheet.col_values(self.grid_id_col)
            self.grid_row_mapping = {}
            
            for i, grid_id in enumerate(grid_id_values[1:], start=2):
                if grid_id in self.grid_coordinates:
                    self.grid_row_mapping[grid_id] = i
            
            print(f"📍 Grid row mapping created")
            
        except Exception as e:
            print(f"⚠️ Error creating row mapping: {e}")
            self.grid_row_mapping = {}
    
    def sheet_update_worker(self):
        """Background worker for Google Sheets updates"""
        while True:
            try:
                if not self.sheet_update_queue.empty():
                    detected_people = self.sheet_update_queue.get()
                    self.perform_sheet_update(detected_people)
                    time.sleep(0.1)  # Small delay to prevent overwhelming the API
                else:
                    time.sleep(0.5)  # Check queue every 0.5 seconds
            except Exception as e:
                print(f"⚠️ Sheet update worker error: {e}")
                time.sleep(1)
    

    def perform_sheet_update(self, detected_people):
        """Perform actual Google Sheets update - only when people are detected"""
        try:
            # Only update if people are detected
            if detected_people > 0:
                current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Only update G22 when people are detected
                if 'G22' in self.grid_row_mapping:
                    row_num = self.grid_row_mapping['G22']
                    
                    batch_updates = [
                        {
                            'range': f'{chr(64 + self.human_detection_col)}{row_num}',
                            'values': [[1]]
                        },
                        {
                            'range': f'{chr(64 + self.timestamp_col)}{row_num}',
                            'values': [[current_timestamp]]
                        },
                        {
                            'range': f'{chr(64 + self.s_latitude1_col)}{row_num}',
                            'values': [[3.1241604064208124]]
                        },
                        {
                            'range': f'{chr(64 + self.s_longtitude1_col)}{row_num}',
                            'values': [[101.66002415090372]]
                        }
                    ]
                    
                    # Perform batch update
                    self.sheet.batch_update(batch_updates)
                    print(f"📊 Updated sheet: G22 detection = 1, People count = {detected_people}, Time = {current_timestamp}")
                    print(f"📍 Updated coordinates: s_latitude1 = 3.1241604064208124, s_longtitude1 = 101.66002415090372")
            # If no people detected, do nothing - preserve existing values
                    
        except Exception as e:
            print(f"⚠️ Google Sheets update failed: {e}")
    
    def get_person_color(self, person_id):
        return (0, 255, 0)
    
    def update_tracking_history(self, person_id, center_point, max_history=50):  # Reduced history
        if person_id in self.track_history:
            self.track_history[person_id].append(center_point)
            if len(self.track_history[person_id]) > max_history:
                self.track_history[person_id] = self.track_history[person_id][-max_history:]
        else:
            self.track_history[person_id] = [center_point]
    
    def draw_tracking_path(self, frame, person_id):
        if person_id not in self.track_history or len(self.track_history[person_id]) < 2:
            return
        
        points = self.track_history[person_id]
        color = self.get_person_color(person_id)
        
        # Draw fewer points for better performance
        step = max(1, len(points) // 20)  # Draw maximum 20 line segments
        for i in range(step, len(points), step):
            thickness = max(1, int(2 * (i / len(points))))
            cv2.line(frame, points[i-step], points[i], color, thickness)
    
    def draw_detection_info(self, frame, box, person_id, confidence):
        x1, y1, x2, y2 = box.astype(int)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        color = self.get_person_color(person_id)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw center point
        cv2.circle(frame, (center_x, center_y), 4, color, -1)
        
        # Draw label
        label = f"Person {person_id} ({confidence:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return center_x, center_y
    
    def create_stats_overlay(self):
        """Pre-create stats overlay to avoid recreation every frame"""
        elapsed_time = time.time() - self.start_time
        current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        stats_text = [
            f"FPS: {current_fps:.1f}",
            f"People Tracked: {len(self.track_history)}",
            f"Frame: {self.frame_count}",
            f"Resolution: {self.width}x{self.height}",
        ]
        
        # Create overlay
        overlay = np.zeros((160, 300, 3), dtype=np.uint8)
        
        for i, text in enumerate(stats_text):
            y_pos = 35 + i * 25
            cv2.putText(overlay, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return overlay
    
    def draw_statistics(self, frame):
        # Only update overlay periodically
        if self.ui_update_counter % self.ui_update_interval == 0 or self.stats_overlay is None:
            self.stats_overlay = self.create_stats_overlay()
        
        # Apply pre-created overlay
        overlay_region = frame[10:170, 10:310]
        blended = cv2.addWeighted(overlay_region, 0.3, self.stats_overlay, 0.7, 0)
        frame[10:170, 10:310] = blended
    
    def draw_instructions(self, frame):
        # Create instructions overlay only once
        if self.instructions_overlay is None:
            instructions = [
                "Controls:",
                "'q' - Quit",
                "'c' - Clear tracks",
                "'s' - Save frame",
                "'r' - Reset stats"
            ]
            
            start_y = self.height - 140
            self.instructions_overlay = np.zeros((130, 190, 3), dtype=np.uint8)
            
            for i, instruction in enumerate(instructions):
                y_pos = 20 + i * 20
                cv2.putText(self.instructions_overlay, instruction, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Apply pre-created overlay
        start_y = self.height - 140
        overlay_region = frame[start_y:start_y + 130, 10:200]
        blended = cv2.addWeighted(overlay_region, 0.3, self.instructions_overlay, 0.7, 0)
        frame[start_y:start_y + 130, 10:200] = blended
    
    def save_frame(self, frame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"detection_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"💾 Frame saved: {filename}")
        return filename
    
    def queue_sheet_update(self, detected_people):
        """Queue Google Sheets update for background processing"""
        if self.sheets_ready and self.sheet_update_queue.qsize() < 3:  # Prevent queue overflow
            self.sheet_update_queue.put(detected_people)
    
    def process_frame(self, frame):
        output_frame = frame.copy()
        detected_people = 0
        
        # Only run detection every few frames
        self.detection_counter += 1
        if self.detection_counter >= self.detection_interval:
            self.detection_counter = 0
            
            # Run YOLO detection with reduced image size for speed
            small_frame = cv2.resize(frame, (320, 240))  # Smaller input for faster processing
            results = self.model.track(small_frame, persist=True, classes=[0], verbose=False)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                # Scale boxes back to original size
                boxes[:, [0, 2]] *= self.width / 320
                boxes[:, [1, 3]] *= self.height / 240
                
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                detected_people = len(track_ids)
                
                for box, person_id, confidence in zip(boxes, track_ids, confidences):
                    if confidence > 0.5:
                        center_x, center_y = self.draw_detection_info(output_frame, box, person_id, confidence)
                        self.update_tracking_history(person_id, (center_x, center_y))
            
            self.last_detected_people = detected_people
        else:
            detected_people = self.last_detected_people
            
            # Draw existing tracking paths
            for person_id in self.track_history:
                self.draw_tracking_path(output_frame, person_id)
        
        return output_frame, detected_people
    
    def run(self):
        print("\n🎯 Starting real-time person detection...")
        print("📝 Controls: 'q'=quit, 'c'=clear tracks, 's'=save frame, 'r'=reset stats")
        print()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame from camera")
                    break
                
                self.frame_count += 1
                self.ui_update_counter += 1
                
                # Process frame for person detection
                output_frame, detected_people = self.process_frame(frame)
                
                # Queue Google Sheets update (non-blocking)
                if self.frame_count % 60 == 0:  # Update every 2 seconds at 30 FPS
                    self.queue_sheet_update(detected_people)
                
                # Draw UI elements
                self.draw_statistics(output_frame)
                self.draw_instructions(output_frame)
                
                # Display frame
                cv2.imshow('Real-time Person Detection & Tracking', output_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("👋 Quitting...")
                    break
                elif key == ord('c'):
                    print("🧹 Clearing tracking history...")
                    self.track_history.clear()
                elif key == ord('s'):
                    self.save_frame(output_frame)
                elif key == ord('r'):
                    print("🔄 Resetting statistics...")
                    self.frame_count = 0
                    self.start_time = time.time()
                    self.stats_overlay = None  # Force recreation
                
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"💥 Error occurred: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        print("🧹 Cleaning up resources...")
        
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        
        if self.frame_count > 0:
            elapsed_time = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            print(f"\n📈 Session Summary:")
            print(f"   • Total frames: {self.frame_count}")
            print(f"   • Average FPS: {avg_fps:.2f}")
            print(f"   • People tracked: {len(self.track_history)}")
            print(f"   • Duration: {elapsed_time:.1f} seconds")
            print(f"   • Saved frames location: {self.output_dir}/")
            print("📊 Sheet data preserved - detection values and timestamps remain intact")

def main():
    try:
        tracker = PersonTracker(camera_index=0)
        tracker.run()
    except Exception as e:
        print(f"💥 Failed to initialize tracker: {e}")
        print("🔧 Troubleshooting tips:")
        print("   • Make sure your camera is connected")
        print("   • Try changing camera_index to 1 or 2")
        print("   • Check if 'credentials.json' and sheet exist")
        print("   • Ensure sheet has columns: grid_id, latitude, longitude, human_detection, timestamp1")

if __name__ == "__main__":
    main()