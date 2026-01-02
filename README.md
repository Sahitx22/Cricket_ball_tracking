# Cricket_ball_tracking
**🏏 Cricket Ball Trajectory Detection & Tracking**

This project detects and tracks a cricket ball in match videos using a YOLO-based object detector, generates a ball trajectory, and outputs per-frame annotations for further analysis or model training.

**The system is optimized to:**

1. Focus only on the pitch region avoiding false detections of cricket ball in logos
2. Reduce false detections (players, crowd, background)
3. Generate both visual and structured data outputs

**📌 Features**

✅ Cricket ball detection using YOLO

✅ Pitch-focused inference (ignores 20% frame width on both sides)

✅ Centroid-based trajectory visualization

✅ Per-frame annotation output (CSV)

✅ Video output with overlays

✅ Works on offline video input

✅ CPU-friendly (no GPU required)

## 📁 Project Structure

```text
Cricket-Ball-Trajectory-Prediction/
├── runs/
│   └── detect/
│       └── train5/
│           └── weights/
│               └── last.pt        # Trained YOLO model
│
├── videos/
│   └── 6.mov                      # Input video
│
├── outputs/
│   ├── trajectory_output.mp4      # Output video with trajectory
│   └── ball_annotations.csv       # Per-frame annotation file
│
├── predict.py                     # Main inference + tracking script
├── requirements.txt
└── README.md
```
**⚙️ Environment Setup**

1️⃣ Create virtual environment
```
  python3 -m venv cricket_env
  source cricket_env/bin/activate
```
2️⃣ Install dependencies
```
  pip install -r requirements.txt
```

Note: The project is tested on Python 3.9.

**▶️ Running Inference**

1️⃣ Place your video
```
videos/6.mov
```
2️⃣ Run the tracking script
```  
python predict.py
```
3️⃣ Outputs generated
```
📹 outputs/trajectory_output.mp4

📄 outputs/ball_annotations.csv
```
Press q to stop playback.

📊 Per-Frame Annotation Format

The CSV file contains one row per video frame:
```
frame_idx,centroid_x,centroid_y,visible
```

Field	Description
```
frame_idx	Frame number
centroid_x	X coordinate of ball centroid
centroid_y	Y coordinate of ball centroid
visible	1 if ball detected, 0 otherwise
```
If the ball is not detected, centroid values are empty.

**🎯 Pitch Region Filtering**

To avoid false detections:
1. The leftmost 20% and rightmost 20% of the frame are ignored
2. Only detections inside the pitch region are considered

This significantly improves robustness by:
1. Ignoring players & audience
2. Focusing inference where the ball is expected

**📈 Trajectory Generation**

1. Ball centroid is stored in a fixed-size queue
2. Consecutive centroids are connected using line segments
3. Produces a clear ball flight trajectory

**🧠 Design Choices**

Why centroid tracking?
1. Lightweight
2. Stable for small objects like cricket balls
3. Easy to extend to velocity / bounce analysis

**Why CSV annotations?**

1. Easy visualization
2. Compatible with ML pipelines
3. Can be used for trajectory prediction models

