from collections import deque
from ultralytics import YOLO
import math
import time
import cv2
import os
import csv

def angle_between_lines(m1, m2=1):
    if m1 != -1/m2:
        angle = math.degrees(math.atan(abs((m2 - m1) / (1 + m1 * m2))))
        return angle
    else:
        return 90.0

class FixedSizeQueue:
    def __init__(self, max_size):
        self.queue = deque(maxlen=max_size)
    
    def add(self, item):
        self.queue.append(item)
    
    def pop(self):
        if len(self.queue) > 0:
            self.queue.popleft()
        
    def clear(self):
        self.queue.clear()

    def get_queue(self):
        return self.queue
    
    def __len__(self):
        return len(self.queue)


# ---------------- MODEL ---------------- #
model_path = os.path.join('runs','detect','train','weights','last.pt')
model = YOLO(model_path)

# ---------------- VIDEO ---------------- #
video_path = os.path.join('videos','10.mov')
cap = cv2.VideoCapture(video_path)

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps_out = cap.get(cv2.CAP_PROP_FPS)
if fps_out == 0 or fps_out is None:
    fps_out = 60

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
os.makedirs("outputs", exist_ok=True)
out = cv2.VideoWriter(
    'outputs/trajectory_output_10.mp4',
    fourcc,
    fps_out,
    (W, H)
)

# ---------------- CSV ANNOTATION FILE ---------------- #
csv_path = "outputs/ball_annotations_10.csv"
csv_file = open(csv_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame_idx", "centroid_x", "centroid_y", "visible"])

# ---------------- PITCH REGION (20% IGNORE EACH SIDE) ---------------- #
LEFT_LIMIT  = int(0.2 * W)
RIGHT_LIMIT = int(0.8 * W)

# ---------------- STATE ---------------- #
centroid_history = FixedSizeQueue(10)
prev_frame_time = 0
frame_idx = 0

# ---------------- LOOP ---------------- #
while True:
    ret, frame = cap.read()
    if not ret:
        break

    new_frame_time = time.time()
    fps = int(1 / max(new_frame_time - prev_frame_time, 1e-6))
    prev_frame_time = new_frame_time

    visible = 0
    centroid_x = ""
    centroid_y = ""

    # YOLO inference
    results = model(frame, conf=0.5, imgsz=1280, verbose=False)
    boxes = results[0].boxes

    if boxes is not None and len(boxes.xyxy) > 0:
        for box in boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.tolist())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # -------- PITCH FILTER -------- #
            if cx < LEFT_LIMIT or cx > RIGHT_LIMIT:
                continue

            centroid_x = cx
            centroid_y = cy
            visible = 1

            centroid_history.add((cx, cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

            break   # only one ball per frame

    # -------- TRAJECTORY -------- #
    pts = list(centroid_history.get_queue())
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i-1], pts[i], (255, 0, 0), 4)

    # -------- DRAW PITCH REGION (DEBUG) -------- #
    cv2.line(frame, (LEFT_LIMIT, 0), (LEFT_LIMIT, H), (0, 255, 0), 1)
    cv2.line(frame, (RIGHT_LIMIT, 0), (RIGHT_LIMIT, H), (0, 255, 0), 1)

    # -------- HUD -------- #
    cv2.putText(frame, f'FPS: {fps}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # -------- WRITE CSV ROW -------- #
    csv_writer.writerow([frame_idx, centroid_x, centroid_y, visible])

    # -------- OUTPUT -------- #
    out.write(frame)
    cv2.imshow('frame', cv2.resize(frame, (1000, 600)))

    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ---------------- #
cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

