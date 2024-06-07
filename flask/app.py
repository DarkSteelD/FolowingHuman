from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import time
import os

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # You can choose 'yolov8n.pt', 'yolov8s.pt', etc., for different sizes

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

def calculate_control_signal(bbox, image_width, image_height, K_p, target_height_ratio):
    # Extract bounding box parameters
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    # Calculate FOV center
    x_FOV = image_width / 2
    y_FOV = image_height / 2

    # Calculate errors
    delta_x = x_center - x_FOV
    delta_y = y_center - y_FOV

    # Calculate control signals for position
    u_x = K_p * delta_y  # Adjust forward/backward based on vertical error
    u_y = K_p * delta_x  # Adjust left/right based on horizontal error

    # Calculate control signals for altitude
    current_height_ratio = bbox_height / image_height
    if current_height_ratio < target_height_ratio:
        # If the person is not fully visible, we need to fly up
        u_z = K_p * (target_height_ratio - current_height_ratio)
    else:
        u_z = 0  # No altitude adjustment needed

    # Calculate control signals for yaw
    u_yaw = K_p * delta_x  # Adjust yaw based on horizontal error

    return u_x, u_y, u_z, u_yaw

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    global frame_count, control_signals
    K_p = 0.1  # Proportional gain
    target_height_ratio = 0.75  # Target height ratio (e.g., 75% of the image height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Use the model to perform inference
        results = model(frame)

        # Extract bounding boxes and class labels
        largest_box = None
        largest_area = 0
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            for box in boxes:
                cls = box.cls  # Class id
                if int(cls) == 0:  # Class '0' is 'person' in COCO dataset
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    conf = box.conf[0]  # Confidence score
                    area = (x2 - x1) * (y2 - y1)
                    if area > largest_area:
                        largest_area = area
                        largest_box = (x1, y1, x2, y2)

                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Person: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate and print control signals if a person is detected
        if largest_box is not None:
            frame_height, frame_width = frame.shape[:2]
            move_x, move_y, move_z, move_yaw = calculate_control_signal(largest_box, frame_width, frame_height, K_p, target_height_ratio)
            control_signals = {'move_x': move_x, 'move_y': move_y, 'move_z': move_z, 'move_yaw': move_yaw}
            # Add the control arrows to the frame
            if move_yaw < 0:
                arrow_text = 'Rotate Left'
            elif move_yaw > 0:
                arrow_text = 'Rotate Right'
            else:
                arrow_text = 'Centered'
            control_text = f"move_x: {move_x:.2f}, move_y: {move_y:.2f}, move_z: {move_z:.2f}, move_yaw: {move_yaw:.2f}"
            cv2.putText(frame, control_text, (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, arrow_text, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            arrow_text = 'No person detected'
            cv2.putText(frame, arrow_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            control_signals = {'move_x': 0, 'move_y': 0, 'move_z': 0, 'move_yaw': 0}

        # Save the captured frame
        frame_filename = f'static/captured_frames/frame_{frame_count:04d}.jpg'
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

        # Stream the frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control_signals')
def get_control_signals():
    global control_signals
    return jsonify(control_signals)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
