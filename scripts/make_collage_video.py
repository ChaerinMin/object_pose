"""Reconstruct multiview_pose_collage.mp4 from the fallback frames directory.

Usage:
    python scripts/make_collage_video.py [--seq-roots <dir> [<dir> ...]] [--fps 10] [--crf 18] [--target-size WxH]

Examples:
    # Single sequence
    python scripts/make_collage_video.py \
        --seq-roots /oscar/data/ssrinath/public/brics-mini/2026-04-10_cylindar1/multisequence000001

    # All cylindar sequences at once (glob-style via shell expansion)
    python scripts/make_collage_video.py \
        --seq-roots /oscar/data/ssrinath/public/brics-mini/2026-04-10_cylindar*/multisequence000001

For each seq-root the script looks for:
    <seq-root>/outputs/object_pose/videos/multiview_pose_collage_frames/*.png

and writes:
    <seq-root>/outputs/object_pose/videos/multiview_pose_collage.mp4

If the frames have inconsistent sizes within a sequence, every frame is resized
to --target-size (default: the most common size among all frames in that sequence).
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image


FRAMES_SUBDIR = "outputs/object_pose/videos/multiview_pose_collage_frames"
OUT_VIDEO_SUBDIR = "outputs/object_pose/videos"
OUT_VIDEO_NAME = "multiview_pose_collage.mp4"


def find_ffmpeg() -> str:
    import shutil
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    # Common module-loaded paths on Oscar
    candidates = glob.glob("/oscar/rt/*/spack/*/ffmpeg-*/bin/ffmpeg")
    if candidates:
        return sorted(candidates)[-1]
    raise RuntimeError(
        "ffmpeg not found. Load it first:  module load ffmpeg"
    )


def make_video(frames: list[Path], out_path: Path, fps: int, crf: int,
               target_size: tuple[int, int] | None, ffmpeg: str) -> None:
    """Write frames to an mp4. Resizes if needed."""
    sizes = Counter(Image.open(f).size for f in frames)
    dominant = target_size or sizes.most_common(1)[0][0]
    needs_resize = any(s != dominant for s in sizes)

    print(f"  {len(frames)} frames | sizes={dict(sizes)} | target={dominant} | resize={needs_resize}")

    if needs_resize:
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, fpath in enumerate(frames):
                img = Image.open(fpath).convert("RGB").resize(dominant, Image.LANCZOS)
                img.save(os.path.join(tmpdir, f"{i:06d}.png"))
            _run_ffmpeg(ffmpeg, os.path.join(tmpdir, "%06d.png"), out_path, fps, crf)
    else:
        frames_dir = frames[0].parent
        _run_ffmpeg(ffmpeg, str(frames_dir / "%06d.png"), out_path, fps, crf)


def _run_ffmpeg(ffmpeg: str, input_pattern: str, out_path: Path, fps: int, crf: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y",
        "-r", str(fps),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR (ffmpeg):\n{result.stderr[-600:]}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build collage MP4 from fallback frames.")
    parser.add_argument("--seq-roots", nargs="+", required=True,
                        help="One or more sequence root directories (multisequenceXXXXXX level).")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--crf", type=int, default=18,
                        help="libx264 CRF quality (lower = better, default 18).")
    parser.add_argument("--target-size", type=str, default=None,
                        help="Force output WxH, e.g. 2560x1440. Default: most common size per sequence.")
    args = parser.parse_args()

    target_size: tuple[int, int] | None = None
    if args.target_size:
        w, h = args.target_size.lower().split("x")
        target_size = (int(w), int(h))

    ffmpeg = find_ffmpeg()
    print(f"Using ffmpeg: {ffmpeg}\n")

    for seq_root in args.seq_roots:
        frames_dir = Path(seq_root) / FRAMES_SUBDIR
        out_path = Path(seq_root) / OUT_VIDEO_SUBDIR / OUT_VIDEO_NAME

        if not frames_dir.is_dir():
            print(f"SKIP (no frames dir): {frames_dir}")
            continue

        frames = sorted(frames_dir.glob("*.png"))
        if not frames:
            print(f"SKIP (empty): {frames_dir}")
            continue

        print(f"{seq_root}")
        make_video(frames, out_path, args.fps, args.crf, target_size, ffmpeg)
        print(f"  -> {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
