import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_POSES = Path(
    "/oscar/home/sharitha/data/users/sharitha/Hands/Handprocess/hand_pose_estimation/outputs/"
    "multiview_object_pose_nvdiffrast/poses/optimized_poses.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot average loss from optimized_poses.json")
    parser.add_argument("--poses", type=Path, default=DEFAULT_POSES)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    poses_path = args.poses
    if not poses_path.exists():
        raise FileNotFoundError(f"Missing poses file: {poses_path}")

    with poses_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError("No pose entries found in JSON.")

    items = []
    for key, val in data.items():
        try:
            idx = int(str(key).split("_")[-1])
        except Exception:
            idx = key
        items.append((idx, float(val.get("loss", 0.0))))

    items.sort(key=lambda x: x[0])
    indices = [i for i, _ in items]
    losses = [l for _, l in items]

    running = []
    total = 0.0
    for i, l in enumerate(losses, start=1):
        total += l
        running.append(total / i)

    avg_loss = total / max(len(losses), 1)

    out_path = args.out or poses_path.parent / "avg_loss.png"

    plt.figure(figsize=(10, 5))
    plt.plot(indices, losses, label="Loss per timestamp", alpha=0.6)
    plt.plot(indices, running, label="Running average", linewidth=2.0)
    plt.axhline(avg_loss, color="red", linestyle="--", label=f"Average = {avg_loss:.6f}")
    plt.xlabel("Timestamp index")
    plt.ylabel("Loss")
    plt.title("Optimization Loss Across Timestamps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
