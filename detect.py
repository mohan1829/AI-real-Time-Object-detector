from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

prev_time = time.time()

while True:
    success, frame = cap.read()

    if not success:
        break

    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    current_time = time.time()
    fps = int(1 / (current_time - prev_time))
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

    cv2.imshow("YOLO Object Detector", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()