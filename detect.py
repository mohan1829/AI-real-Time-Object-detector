from ultralytics import YOLO
import cv2
import time
import os
import pyttsx3
import csv 
from datetime import datetime

# Load YOLO model
model = YOLO("yolov8n.pt")

# Voice engine
engine = pyttsx3.init()

last_announced = ""
last_time = 0

# Webcam
cap = cv2.VideoCapture(0)

# Create detections folder
if not os.path.exists("detections"):
    os.makedirs("detections")
    
if not os.path.exists("detection_log.csv"):
    with open("detection_log.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Object"])

prev_time = time.time()

while True:
    success, frame = cap.read()

    if not success:
        break

    # Run YOLO
    results = model(frame, verbose=False)

    # Voice Alert
    if len(results[0].boxes) > 0:
        first_box = results[0].boxes[0]
        detected_name = model.names[int(first_box.cls[0])]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open("detection_log.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, detected_name])

        if detected_name != last_announced or time.time() - last_time > 5:
            engine.say(f"{detected_name} detected")
            engine.runAndWait()

            last_announced = detected_name
            last_time = time.time()

        # Save image
        filename = f"detections/detection_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)

    # Count objects
    object_counts = {}

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]

        if name in object_counts:
            object_counts[name] += 1
        else:
            object_counts[name] = 1

    # Draw boxes
    annotated_frame = results[0].plot()

    # FPS
    current_time = time.time()

    if current_time - prev_time > 0:
        fps = int(1 / (current_time - prev_time))
    else:
        fps = 0

    prev_time = current_time

    cv2.putText(
        annotated_frame,
        f"FPS: {fps}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show counts
    y = 80

    for name, count in object_counts.items():
        cv2.putText(
            annotated_frame,
            f"{name}: {count}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )
        y += 30

    # Display frame
    cv2.imshow("YOLO Object Detector", annotated_frame)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()