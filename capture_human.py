import cv2
from ultralytics import YOLO
import time
import os

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # You can choose 'yolov8n.pt', 'yolov8s.pt', etc., for different sizes

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # Change this if your webcam is not the first video device

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# Ensure the directory to save frames exists
if not os.path.exists('captured_frames'):
    os.makedirs('captured_frames')

print("Press 'Ctrl+C' to stop capturing.")

frame_count = 0

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

# Parameters for control signal calculation
K_p = 0.1  # Proportional gain
target_height_ratio = 0.75  # Target height ratio (e.g., 75% of the image height)

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image")
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

        # Print coordinates and calculate move for the largest box
        if largest_box is not None:
            print(f"Largest box coordinates: {largest_box}")
            frame_height, frame_width = frame.shape[:2]
            move_x, move_y, move_z, move_yaw = calculate_control_signal(largest_box, frame_width, frame_height, K_p, target_height_ratio)
            print(f"Control Signals: move_x = {move_x:.2f}, move_y = {move_y:.2f}, move_z = {move_z:.2f}, move_yaw = {move_yaw:.2f}")
            # Add the text to the frame
            control_text = f"move_x: {move_x:.2f}, move_y: {move_y:.2f}, move_z: {move_z:.2f}, move_yaw: {move_yaw:.2f}"
            cv2.putText(frame, control_text, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Save the captured frame
        frame_filename = os.path.join('captured_frames', f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        print(f'Captured {frame_filename}')
        frame_count += 1

        # Wait for 1 second before capturing the next frame
        time.sleep(1)

except KeyboardInterrupt:
    print("Interrupted by user")

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
