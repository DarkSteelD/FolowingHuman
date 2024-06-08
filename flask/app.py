from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import time
import os

app = Flask(__name__)

# Load the YOLOv8 model trained for person, fire, and smoke detection
model = YOLO('yolov8n.pt')  # Use your custom-trained model file

# Try to open a connection to the available webcams
def open_camera():
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Opened video device {i}")
            return cap
    print("Error: Could not open any video device.")
    exit()

cap = open_camera()

# Ensure the directory to save frames exists
if not os.path.exists('static/captured_frames'):
    os.makedirs('static/captured_frames')

frame_count = 0
control_signals = {'move_x': 0, 'move_y': 0, 'move_z': 0, 'move_yaw': 0}
fire_detected = False
smoke_detected = False

# Initialize tracker
tracker = cv2.TrackerCSRT_create()  # You can choose different trackers like KCF, CSRT, etc.
tracking = False
bbox = None

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    global frame_count, control_signals, fire_detected, smoke_detected, tracking, tracker, bbox
    target_height_ratio = 0.75  # Target height ratio (e.g., 75% of the image height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Use the model to perform inference if not tracking
        if not tracking:
            results = model(frame)

            # Extract bounding boxes and class labels
            largest_box = None
            largest_area = 0
            fire_detected = False
            smoke_detected = False

            for r in results:
                boxes = r.boxes  # Boxes object for bbox outputs
                for box in boxes:
                    cls = box.cls  # Class id
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    conf = box.conf[0]  # Confidence score
                    area = (x2 - x1) * (y2 - y1)
                    
                    if int(cls) == 0:  # Class '0' is 'person' in COCO dataset
                        if area > largest_area:
                            largest_area = area
                            largest_box = (x1, y1, x2, y2)
                        
                        # Draw the bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Person: {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    elif int(cls) == 74:  # Class for fire (adjust as necessary for your custom model)
                        fire_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f"Fire: {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    elif int(cls) == 75:  # Class for smoke (adjust as necessary for your custom model)
                        smoke_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        label = f"Smoke: {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # If a person is detected, initialize the tracker
            if largest_box is not None:
                bbox = (largest_box[0], largest_box[1], largest_box[2] - largest_box[0], largest_box[3] - largest_box[1])
                tracker.init(frame, bbox)
                tracking = True
        else:
            # Update the tracker
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"Tracking Person"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                control_signals = {'move_x': 1, 'move_y': 1, 'move_z': 1, 'move_yaw': 1}  # Fake PID commands
            else:
                tracking = False
                control_signals = {'move_x': 0, 'move_y': 0, 'move_z': 0, 'move_yaw': 0}

        # Add fire and smoke alerts to the frame
        if fire_detected:
            cv2.putText(frame, "Fire Detected!", (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if smoke_detected:
            cv2.putText(frame, "Smoke Detected!", (10, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        frame_count += 1

        # Stream the frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control_signals')
def get_control_signals():
    global control_signals
    return jsonify(control_signals)

@app.route('/alerts')
def get_alerts():
    global fire_detected, smoke_detected
    return jsonify({'fire_detected': fire_detected, 'smoke_detected': smoke_detected})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
