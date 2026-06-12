import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime

DEVICE = "cuda"
THRESHOLD = 8        # pixels of movement to count as active
HISTORY = 5          # compare against N frames ago
SOURCE = 0           # 0 = webcam, or rtsp url, or video file path

model = YOLO("yolov8n-pose.pt")  # downloads automatically

# stores last N keypoints per track id
keypoint_history = defaultdict(list)

def get_movement(kp_prev, kp_curr):
    # kp shape: (17, 3) -> x, y, confidence
    prev = np.array([[k[0], k[1]] for k in kp_prev])
    curr = np.array([[k[0], k[1]] for k in kp_curr])
    conf = np.array([k[2] for k in kp_curr])
    # only use keypoints with high confidence
    mask = conf > 0.5
    if mask.sum() == 0:
        return 0
    dists = np.linalg.norm(curr[mask] - prev[mask], axis=1)
    return dists.mean()

def run(source=SOURCE):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Failed to open source")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        results = model.track(frame, persist=True, verbose=False, device=DEVICE)

        if results[0].keypoints is None:
            cv2.imshow("Pose Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        boxes = results[0].boxes
        keypoints = results[0].keypoints.data.cpu().numpy()  # (N, 17, 3)

        active, idle = 0, 0

        for i, kp in enumerate(keypoints):
            # get track id
            if boxes.id is None:
                continue
            track_id = int(boxes.id[i])
            keypoint_history[track_id].append(kp)

            if len(keypoint_history[track_id]) > HISTORY + 1:
                keypoint_history[track_id].pop(0)

            # get bounding box
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])

            if len(keypoint_history[track_id]) < HISTORY:
                label = "warming up"
                color = (128, 128, 128)
            else:
                movement = get_movement(
                    keypoint_history[track_id][0],
                    keypoint_history[track_id][-1]
                )
                if movement < THRESHOLD:
                    label = f"idle ({movement:.1f})"
                    color = (0, 0, 255)
                    idle += 1
                else:
                    label = f"active ({movement:.1f})"
                    color = (0, 255, 0)
                    active += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID{track_id}: {label}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log = f"{timestamp} | Active: {active} | Idle: {idle}"
        cv2.putText(frame, log, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        print(log)

        with open("runs/activity_log.txt", "a") as f:
            f.write(log + "\n")

        cv2.imshow("Pose Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()