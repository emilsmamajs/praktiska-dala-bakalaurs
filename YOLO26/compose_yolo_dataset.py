import json
import os
from pathlib import Path

SCRIPT_DIR    = Path(__file__).resolve().parent
BAKALAURS_DIR = SCRIPT_DIR.parent / "BAKALAURS_DATASET"
OUT_DIR       = SCRIPT_DIR / "yolo_dataset"

CLASS_MAP   = {"not_occluded": 0, "occluded": 1}
CLASS_NAMES = ["not_occluded", "occluded"]

DOCKER_DATA_PATH = "/data/YOLO26/yolo_dataset"


def main():
    clamp = lambda v: max(0.0, min(1.0, v))

    for split in ("train", "val"):
        for class_name, class_id in CLASS_MAP.items():
            src_dir = BAKALAURS_DIR / split / class_name

            img_out = OUT_DIR / "images" / split
            lbl_out = OUT_DIR / "labels" / split

            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            for img_path in sorted(src_dir.glob("*.jpg")):
                json_path = img_path.with_suffix(".json")

                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)

                img_w = data["width"]
                img_h = data["height"]

                label_lines = []
                for obj in data["objects"]:
                    b = obj["bbox"]
                    x_c = clamp((b["xmin"] + b["xmax"]) / 2.0 / img_w)
                    y_c = clamp((b["ymin"] + b["ymax"]) / 2.0 / img_h)
                    w   = clamp((b["xmax"] - b["xmin"]) / img_w)
                    h   = clamp((b["ymax"] - b["ymin"]) / img_h)
                    label_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

                lbl_file = lbl_out / (img_path.stem + ".txt")
                lbl_file.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))

                dst_img = img_out / img_path.name
                if not dst_img.exists():
                    rel = os.path.relpath(img_path.resolve(), dst_img.parent)
                    os.symlink(rel, dst_img)

    (OUT_DIR / "data.yaml").write_text(
        f"path: {DOCKER_DATA_PATH}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )


if __name__ == "__main__":
    main()
