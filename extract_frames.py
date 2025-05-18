import cv2
import os
from pathlib import Path

def video_to_frames(video_path, output_dir):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = output_dir / f"frame_{frame_idx:04d}.jpg"
        cv2.imwrite(str(frame_filename), frame)
        frame_idx += 1
    cap.release()
    print(f"✅ Zapisano {frame_idx} klatek do folderu: {output_dir}")

video_to_frames("data/raw/video3.mpg", "data/frames/film3")
