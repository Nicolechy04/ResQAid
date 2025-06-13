"""
Simplified ESP32 Person Detection and Tracking - Green Color Only
ESP32 streams video over WiFi, this script processes it for person detection
"""
import cv2 # type: ignore
import numpy as np
from ultralytics import YOLO # type: ignore
import time
import requests # type: ignore
from datetime import datetime

class ESP32PersonTracker:
    def __init__(self, esp32_ip="192.168.1.100", esp32_port=81, model_path="yolov8n.pt"):
        """Initialize ESP32 Person Tracker with green color only"""
        print("🚀 Initializing ESP32 Person Tracker...")
        
        # ESP32 camera stream URLs
        self.esp32_ip = esp32_ip
        self.esp32_port = esp32_port
        self.stream_url = f"http://{esp32_ip}:{esp32_port}/stream"
        self.snapshot_url = f"http://{esp32_ip}:{esp32_port}/capture"
        
        print(f"📡 ESP32 Stream URL: {self.stream_url}")
        
        # Load YOLO model
        print("📦 Loading YOLOv8 model...")
        self.model = YOLO(model_path)
        print("✅ Model loaded successfully!")
        
        # Test ESP32 connection
        if not self.test_esp32_connection():
            raise Exception("❌ Could not connect to ESP32 camera")
        
        # Initialize video capture
        print("📹 Connecting to ESP32 camera stream...")
        self.cap = cv2.VideoCapture(self.stream_url)
        
        if not self.cap.isOpened():
            print("🔄 Using alternative connection method...")
            self.use_requests = True
        else:
            self.use_requests = False
            print("✅ ESP32 camera connected successfully!")
        
        # Initialize tracking variables
        self.track_history = {}
        self.frame_count = 0
        self.start_time = time.time()
        
        # Green color only (BGR format)
        self.green_color = (0, 255, 0)
    
    def test_esp32_connection(self):
        """Test if ESP32 is accessible"""
        try:
            response = requests.get(f"http://{self.esp32_ip}:{self.esp32_port}/", timeout=5)
            if response.status_code == 200:
                print("✅ ESP32 connection successful!")
                return True
        except:
            pass
        
        print("❌ Could not connect to ESP32")
        return False
    
    def get_frame_from_esp32(self):
        """Get frame from ESP32 using requests"""
        try:
            response = requests.get(self.snapshot_url, timeout=1)
            if response.status_code == 200:
                nparr = np.frombuffer(response.content, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return True, frame
        except:
            pass
        return False, None
    
    def update_tracking_history(self, person_id, center_point, max_history=50):
        """Update tracking history for a person"""
        if person_id in self.track_history:
            self.track_history[person_id].append(center_point)
            if len(self.track_history[person_id]) > max_history:
                self.track_history[person_id] = self.track_history[person_id][-max_history:]
        else:
            self.track_history[person_id] = [center_point]
    
    def draw_tracking_path(self, frame, person_id):
        """Draw the green tracking path for a person"""
        if person_id not in self.track_history or len(self.track_history[person_id]) < 2:
            return
        
        points = self.track_history[person_id]
        
        # Draw green tracking path
        for i in range(1, len(points)):
            thickness = max(1, int(3 * (i / len(points))))
            cv2.line(frame, points[i-1], points[i], self.green_color, thickness)
    
    def draw_detection_info(self, frame, box, person_id, confidence):
        """Draw green bounding box and information"""
        x1, y1, x2, y2 = box.astype(int)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Draw green bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.green_color, 2)
        
        # Draw green center point
        cv2.circle(frame, (center_x, center_y), 4, self.green_color, -1)
        
        # Draw ID and confidence with green background
        label = f"Person {person_id} ({confidence:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), self.green_color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return center_x, center_y
    
    def draw_statistics(self, frame):
        """Draw performance statistics"""
        elapsed_time = time.time() - self.start_time
        current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        stats_text = [
            f"FPS: {current_fps:.1f}",
            f"People: {len(self.track_history)}",
            f"Frame: {self.frame_count}"
        ]
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw statistics in white text
        for i, text in enumerate(stats_text):
            y_pos = 30 + i * 20
            cv2.putText(frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_frame(self, frame):
        """Process frame for person detection and tracking"""
        if frame is None:
            return None, 0
        
        # Run YOLO detection and tracking (class 0 = person)
        results = self.model.track(frame, persist=True, classes=[0], verbose=False)
        
        output_frame = frame.copy()
        detected_people = 0
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            detected_people = len(track_ids)
            
            for box, person_id, confidence in zip(boxes, track_ids, confidences):
                if confidence > 0.5:  # Confidence threshold
                    center_x, center_y = self.draw_detection_info(
                        output_frame, box, person_id, confidence
                    )
                    
                    self.update_tracking_history(person_id, (center_x, center_y))
                    self.draw_tracking_path(output_frame, person_id)
        
        return output_frame, detected_people
    
    def run(self):
        """Main loop for real-time detection"""
        print(f"\n🎯 Starting ESP32 person detection...")
        print(f"📡 ESP32 Camera: {self.esp32_ip}:{self.esp32_port}")
        print("📝 Controls: 'q'=quit, 'c'=clear tracks")
        
        try:
            while True:
                # Get frame from ESP32
                if self.use_requests:
                    ret, frame = self.get_frame_from_esp32()
                else:
                    ret, frame = self.cap.read()
                
                if not ret:
                    print("❌ Failed to read frame from ESP32")
                    continue
                
                self.frame_count += 1
                
                # Process frame for person detection
                output_frame, detected_people = self.process_frame(frame)
                
                if output_frame is not None:
                    # Draw statistics
                    self.draw_statistics(output_frame)
                    
                    # Display the frame
                    cv2.imshow('ESP32 Person Detection - Green Tracking', output_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("👋 Quitting...")
                    break
                elif key == ord('c'):
                    print("🧹 Clearing tracking history...")
                    self.track_history.clear()
        
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
        
        if self.frame_count > 0:
            elapsed_time = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            print(f"\n📈 Session Summary:")
            print(f"   • Total frames: {self.frame_count}")
            print(f"   • Average FPS: {avg_fps:.2f}")
            print(f"   • People tracked: {len(self.track_history)}")

def main():
    """Main function to run the ESP32 person tracker"""
    try:
        # CHANGE THIS TO YOUR ESP32 IP ADDRESS
        esp32_ip = "192.168.1.100"
        
        print("🔧 ESP32 Setup:")
        print("1. Flash ESP32 with camera server code")
        print("2. Connect ESP32 to WiFi")
        print("3. Update esp32_ip variable above")
        print(f"4. Current ESP32 IP: {esp32_ip}")
        print("\n" + "="*40 + "\n")
        
        # Create and run the tracker
        tracker = ESP32PersonTracker(
            esp32_ip=esp32_ip,
            esp32_port=81,
            model_path="yolov8n.pt"
        )
        tracker.run()
        
    except Exception as e:
        print(f"💥 Failed to initialize ESP32 tracker: {e}")

if __name__ == "__main__":
    main()