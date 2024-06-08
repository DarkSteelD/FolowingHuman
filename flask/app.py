from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load the YOLOv8 model trained for person detection (COCO dataset)
model = YOLO('yolov8n.pt')  # Use the YOLOv8 model file

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

frame_count = 0
control_signals = {'move_x': 0, 'move_y': 0}
tracking = False
bbox = None

@app.route('/')
def index():
    return render_template('index.html')

def calculate_control_signal(bbox, image_width, image_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    bbox_height = y_max - y_min

    # Calculate FOV center
    x_FOV = image_width / 2
    y_FOV = image_height / 2

    # Control signals to center the bounding box
    move_x = 0
    move_y = 0

    if x_center < x_FOV - 20:
        move_y = -1  # Move left
    elif x_center > x_FOV + 20:
        move_y = 1  # Move right

    if bbox_height < 0.75 * image_height:
        move_x = 1  # Move closer
    elif bbox_height > 0.75 * image_height:
        move_x = -1  # Move back

    return move_x, move_y

def gen():
    global frame_count, control_signals, tracking, bbox

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)

        # Extract bounding boxes and class labels
        largest_box = None
        largest_area = 0

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

        # Calculate control signals if a person is detected
        if largest_box is not None:
            frame_height, frame_width = frame.shape[:2]
            move_x, move_y = calculate_control_signal(largest_box, frame_width, frame_height)
            control_signals = {'move_x': move_x, 'move_y': move_y}

            control_text = f"move_x: {move_x}, move_y: {move_y}"
            cv2.putText(frame, control_text, (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
