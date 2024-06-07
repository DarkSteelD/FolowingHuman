import cv2
import os
import time

# Create a directory to save the frames
if not os.path.exists('captured_frames'):
    os.makedirs('captured_frames')

# Open a connection to the webcam (0 is the default ID for the first camera)
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

print("Press 'q' to stop capturing.")

frame_count = 0

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image")
            break

        # Save the captured frame
        frame_filename = f'captured_frames/frame_{frame_count:04d}.jpg'
        cv2.imwrite(frame_filename, frame)
        print(f'Captured {frame_filename}')
        frame_count += 1

        # Wait for 1 second
        time.sleep(1)

        # Check if 'q' is pressed to stop capturing
        if  0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
