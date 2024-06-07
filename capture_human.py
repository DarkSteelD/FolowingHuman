import cv2
from ultralytics import YOLO
import time
import os

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # You can choose 'yolov8n.pt', 'yolov8s.pt', etc., for different sizes

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# Ensure the directory to save frames exists
if not os.path.exists('captured_frames'):
    os.makedirs('captured_frames')

print("Press 'Ctrl+C' to stop capturing.")

frame_count = 0

def calculate_center_move(box, frame_width, frame_height):
    x1, y1, x2, y2 = box
    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2

    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2

    move_x = box_center_x - frame_center_x
    move_y = box_center_y - frame_center_y

    return move_x, move_y

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
            move_x, move_y = calculate_center_move(largest_box, frame_width, frame_height)
            print(f"Move camera by x: {move_x}, y: {move_y} to center the largest frame")
            # Add the text to the frame
            move_text = f"Move camera by x: {move_x:.2f}, y: {move_y:.2f} to center the largest frame"
            cv2.putText(frame, move_text, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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
