from ultralytics import YOLO
import cv2
import time
import os
import pyttsx3
import csv
from datetime import datetime
import smtplib
from email.message import EmailMessage
import streamlit as st 
import pandas as pd 
import os 
from PIL import Image
# ----------------------------
# YOLO Model
# ----------------------------
model = YOLO("yolov8n.pt")

# ----------------------------
# Voice Engine
# ----------------------------
engine = pyttsx3.init()

last_announced = ""
last_voice_time = 0

# ----------------------------
# Email Settings
# ----------------------------
SENDER_EMAIL = "rajamohan@gmail.com"
APP_PASSWORD = "vengeance2022"

last_email_time = 0

def send_email_alert(object_name):
    msg = EmailMessage()

    msg["Subject"] = f"YOLO Alert: {object_name} Detected"
    msg["From"] = SENDER_EMAIL
    msg["To"] = SENDER_EMAIL

    msg.set_content(
        f"Alert!\n\nDetected Object: {object_name}\n\nTime: {datetime.now()}"
    )

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)

        print("Email sent!")

    except Exception as e:
        print("Email Error:", e)

# ----------------------------
# Webcam
# ----------------------------
cap = cv2.VideoCapture(0)

# ----------------------------
# Create Folders
# ----------------------------
if not os.path.exists("detections"):
    os.makedirs("detections")

# ----------------------------
# Create CSV
# ----------------------------
if not os.path.exists("detection_log.csv"):
    with open("detection_log.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Object"])

prev_time = time.time()

# ----------------------------
# Main Loop
# ----------------------------
while True:

    success, frame = cap.read()

    if not success:
        break

    results = model(frame, verbose=False)

    object_counts = {}

    if len(results[0].boxes) > 0:

        first_box = results[0].boxes[0]

        detected_name = model.names[
            int(first_box.cls[0])
        ]

        # Voice Alert
        if (
            detected_name != last_announced
            or time.time() - last_voice_time > 5
        ):

            engine.say(f"{detected_name} detected")
            engine.runAndWait()

            last_announced = detected_name
            last_voice_time = time.time()

        # Email Alert every 30 seconds
        if (
            detected_name == "person"
            and time.time() - last_email_time > 30
        ):

            send_email_alert(detected_name)
            last_email_time = time.time()

        # Save Image
        filename = (
            f"detections/"
            f"detection_{int(time.time())}.jpg"
        )

        cv2.imwrite(filename, frame)

        # CSV Log
        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with open(
            "detection_log.csv",
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [timestamp, detected_name]
            )

    # Object Count
    for box in results[0].boxes:

        cls_id = int(box.cls[0])

        name = model.names[cls_id]

        if name in object_counts:
            object_counts[name] += 1
        else:
            object_counts[name] = 1

    # Draw Boxes
    annotated_frame = results[0].plot()

    # FPS
    current_time = time.time()

    fps = int(
        1 / (current_time - prev_time)
    ) if current_time != prev_time else 0

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

    # Display Object Counts
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

    cv2.imshow(
        "YOLO Smart Surveillance System",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()