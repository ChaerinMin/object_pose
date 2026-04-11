"""
For each output_* folder:
  1. Compute duration from first to last collage image (by file mtime) and save as timing.txt
  2. If videos/ has no video file, create one from collage images
"""

import os
import glob
import re
from datetime import datetime
import cv2
import numpy as np

WORKSPACE = "/oscar/data/ssrinath/users/cmin5/object_pose"
FPS = 10  # frames per second for generated video


def get_sorted_collages(collages_dir):
    images = glob.glob(os.path.join(collages_dir, "*.jpg"))
    images += glob.glob(os.path.join(collages_dir, "*.png"))

    def sort_key(p):
        m = re.search(r"(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0

    return sorted(images, key=sort_key)


def save_timing(output_dir, images):
    first_mtime = os.path.getmtime(images[0])
    last_mtime = os.path.getmtime(images[-1])

    first_dt = datetime.fromtimestamp(first_mtime)
    last_dt = datetime.fromtimestamp(last_mtime)
    duration_sec = last_mtime - first_mtime
    duration_hr = duration_sec / 3600.0

    txt_path = os.path.join(output_dir, "timing.txt")
    with open(txt_path, "w") as f:
        f.write(f"First image : {os.path.basename(images[0])}\n")
        f.write(f"  saved at  : {first_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Last image  : {os.path.basename(images[-1])}\n")
        f.write(f"  saved at  : {last_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total frames: {len(images)}\n")
        f.write(f"Duration    : {duration_hr:.2f} hr ({duration_sec:.1f} sec)\n")

    print(f"  [timing] {duration_hr:.2f} hr over {len(images)} frames -> {txt_path}")


def make_video(output_dir, images):
    videos_dir = os.path.join(output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    out_path = os.path.join(videos_dir, "collage_timelapse.mp4")

    # Read first image to get dimensions
    first = cv2.imread(images[0])
    if first is None:
        print(f"  [video]  FAILED: could not read {images[0]}")
        return
    h, w = first.shape[:2]
    # Ensure dimensions are even (required by some codecs)
    w = w - (w % 2)
    h = h - (h % 2)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (w, h))

    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  [video]  warning: skipping unreadable frame {img_path}")
            continue
        frame = cv2.resize(frame, (w, h))
        writer.write(frame)

    writer.release()
    print(f"  [video]  created -> {out_path}")


def has_video(videos_dir):
    if not os.path.isdir(videos_dir):
        return False
    for f in os.listdir(videos_dir):
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            return True
    return False


def process_output_dir(output_dir):
    print(f"\nProcessing: {output_dir}")
    collages_dir = os.path.join(output_dir, "collages")
    videos_dir = os.path.join(output_dir, "videos")

    if not os.path.isdir(collages_dir):
        print("  [skip] no collages/ directory")
        return

    images = get_sorted_collages(collages_dir)
    if len(images) < 2:
        print(f"  [skip] not enough images ({len(images)})")
        return

    save_timing(output_dir, images)

    if has_video(videos_dir):
        print(f"  [video]  already exists, skipping")
    else:
        make_video(output_dir, images)


def main():
    output_dirs = sorted(glob.glob(os.path.join(WORKSPACE, "output_*")))
    output_dirs = [d for d in output_dirs if os.path.isdir(d)]

    if not output_dirs:
        print("No output_* directories found.")
        return

    for d in output_dirs:
        process_output_dir(d)

    print("\nDone.")


if __name__ == "__main__":
    main()
