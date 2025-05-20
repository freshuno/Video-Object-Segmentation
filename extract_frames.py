import cv2
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Extract frames from video.")
parser.add_argument("video_path", type=str, help="Ścieżka do pliku wideo")
parser.add_argument("output_dir", type=str, help="Ścieżka do katalogu wyjściowego")
args = parser.parse_args()

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

video_to_frames(args.video_path, args.output_dir)
