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
if not os.path.exists('captured_frames'):
    os.makedirs('captured_frames')
print("Press 'Ctrl+C' to stop capturing.")
frame_count = 0
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
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            for box in boxes:
                cls = box.cls  # Class id
                if int(cls) == 0:  # Class '0' is 'person' in COCO dataset
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    conf = box.conf[0]  # Confidence score
                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Person: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the output frame
        frame_filename = f'captured_frames/frame_{frame_count:04d}.jpg'
        cv2.imwrite(frame_filename, frame)
        time.sleep(1)
        #cv2.imshow("Frame", frame)
        frame_count += 1
        # Check if 'q' is pressed to stop capturing
        if 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
