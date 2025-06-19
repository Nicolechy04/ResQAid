"""
Real-time Person Detection and Tracking using YOLOv8
Run this script in VS Code with your webcam connected
"""

import cv2 # type: ignore
import numpy as np
from ultralytics import YOLO # type: ignore
import time
import os
from datetime import datetime

# Manages webcam input, loads the YOLOv8 model, performs detection/tracking, and shows results with tracking paths and statistics.
# camera_index (int): Camera index (0 for default camera)
# model_path (str): Path to YOLO model
class PersonTracker:
    def __init__(self, camera_index=0, model_path="yolov8n.pt"):
        print("🚀 Initializing Person Tracker...")
        
        # Load YOLO model
        print("📦 Loading YOLOv8 model...")
        self.model = YOLO(model_path)
        print("✅ Model loaded successfully!")
        
        # Initialize camera
        print(f"📹 Connecting to camera {camera_index}...")
        self.cap = cv2.VideoCapture(camera_index)
        print(f"🔍 Camera backend in use: {self.cap.getBackendName()}")
        print(f"🔍 Camera properties: {self.cap.get(cv2.CAP_PROP_BACKEND)}")
        
        if not self.cap.isOpened():
            raise Exception(f"❌ Could not open camera {camera_index}")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual camera properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Camera initialized: {self.width}x{self.height} at {self.fps:.1f} FPS")
        
        # Initialize tracking variables
        self.track_history = {}
        self.frame_count = 0
        self.start_time = time.time()
        
        self.person_colors = {
            1: (0, 255, 0),
        }
    
    """Get color for a specific person ID"""
    def get_person_color(self, person_id):
        return (0, 255, 0)  # Always return green (BGR format)
    
    def update_tracking_history(self, person_id, center_point, max_history=100):
        """Update tracking history for a person"""
        if person_id in self.track_history:
            self.track_history[person_id].append(center_point)
            # Limit history to prevent memory issues
            if len(self.track_history[person_id]) > max_history:
                self.track_history[person_id] = self.track_history[person_id][-max_history:]
        else:
            self.track_history[person_id] = [center_point]
    
    """Draw the tracking path for a person"""
    def draw_tracking_path(self, frame, person_id):
        if person_id not in self.track_history or len(self.track_history[person_id]) < 2:
            return
        
        points = self.track_history[person_id]
        color = self.get_person_color(person_id)
        
        # Draw path with fading effect
        for i in range(1, len(points)):
            # Calculate thickness based on recency (newer points are thicker)
            thickness = max(1, int(4 * (i / len(points))))
            cv2.line(frame, points[i-1], points[i], color, thickness)
    
    def draw_detection_info(self, frame, box, person_id, confidence):
        """Draw bounding box and information for a detected person"""
        x1, y1, x2, y2 = box.astype(int)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        color = self.get_person_color(person_id)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw center point
        cv2.circle(frame, (center_x, center_y), 4, color, -1)
        
        # Draw ID and confidence
        label = f"Person {person_id} ({confidence:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Draw background for text
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return center_x, center_y
    
    def draw_statistics(self, frame):
        """Draw performance statistics on the frame"""
        elapsed_time = time.time() - self.start_time
        current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Statistics text
        stats_text = [
            f"FPS: {current_fps:.1f}",
            f"People Tracked: {len(self.track_history)}",
            f"Frame: {self.frame_count}",
            f"Resolution: {self.width}x{self.height}"
        ]
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw statistics
        for i, text in enumerate(stats_text):
            y_pos = 35 + i * 25
            cv2.putText(frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_instructions(self, frame):
        """Draw usage instructions on the frame"""
        instructions = [
            "Controls:",
            "'q' - Quit",
            "'c' - Clear tracks", 
            "'s' - Save frame",
            "'r' - Reset stats"
        ]
        
        start_y = self.height - 140
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, start_y - 20), (200, self.height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw instructions
        for i, instruction in enumerate(instructions):
            y_pos = start_y + i * 20
            cv2.putText(frame, instruction, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def save_frame(self, frame):
        """Save current frame with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"detection_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"💾 Frame saved: {filename}")
        return filename
    
    def process_frame(self, frame):
        """Process a single frame for person detection and tracking"""
        # Run YOLO detection and tracking
        results = self.model.track(frame, persist=True, classes=[0], verbose=False)
        
        output_frame = frame.copy()
        detected_people = 0
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            detected_people = len(track_ids)
            
            # Process each detected person
            for box, person_id, confidence in zip(boxes, track_ids, confidences):
                if confidence > 0.5:  # Filter by confidence threshold
                    # Draw detection info and get center point
                    center_x, center_y = self.draw_detection_info(
                        output_frame, box, person_id, confidence
                    )
                    
                    # Update tracking history
                    self.update_tracking_history(person_id, (center_x, center_y))
                    
                    # Draw tracking path
                    self.draw_tracking_path(output_frame, person_id)
        
        return output_frame, detected_people
    
    def run(self):
        """Main loop for real-time detection"""
        print("\n🎯 Starting real-time person detection...")
        print("📝 Controls: 'q'=quit, 'c'=clear tracks, 's'=save frame, 'r'=reset stats")
        print("🔴 Press any key in the video window to start...\n")
        
        try:
            while True:
                # Read frame from camera
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame from camera")
                    break
                
                self.frame_count += 1
                
                # Process frame for person detection
                output_frame, detected_people = self.process_frame(frame)
                
                # Draw statistics and instructions
                self.draw_statistics(output_frame)
                self.draw_instructions(output_frame)
                
                # Display the frame
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
                    saved_file = self.save_frame(output_frame)
                    print(f"💾 Frame saved: {saved_file}")
                elif key == ord('r'):
                    print("🔄 Resetting statistics...")
                    self.frame_count = 0
                    self.start_time = time.time()
                
                # Print periodic updates
                if self.frame_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    print(f"📊 Processed {self.frame_count} frames, "
                          f"FPS: {fps:.1f}, People: {len(self.track_history)}")
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        
        except Exception as e:
            print(f"💥 Error occurred: {e}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up resources...")
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Print session summary
        if self.frame_count > 0:
            elapsed_time = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            print(f"\n📈 Session Summary:")
            print(f"   • Total frames: {self.frame_count}")
            print(f"   • Average FPS: {avg_fps:.2f}")
            print(f"   • People tracked: {len(self.track_history)}")
            print(f"   • Duration: {elapsed_time:.1f} seconds")
            print(f"   • Saved frames location: {self.output_dir}/")

def main():
    """Main function to run the person tracker"""
    try:
        # Create and run the tracker
        tracker = PersonTracker(camera_index=0)  # Change to 1, 2, etc. for other cameras
        tracker.run()
        
    except Exception as e:
        print(f"💥 Failed to initialize tracker: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("   • Make sure your camera is connected")
        print("   • Close other applications using the camera")
        print("   • Try changing camera_index to 1 or 2")
        print("   • Check if camera permissions are granted")

if __name__ == "__main__":
    main()