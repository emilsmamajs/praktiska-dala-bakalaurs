import json
import os
from pathlib import Path

SCRIPT_DIR    = Path(__file__).resolve().parent
BAKALAURS_DIR = SCRIPT_DIR.parent / "BAKALAURS_DATASET"
OUT_DIR       = SCRIPT_DIR / "coco_dataset"

CLASS_MAP = {"not_occluded": 0, "occluded": 1}

CATEGORIES = [
    {"id": 0, "name": "not_occluded"},
    {"id": 1, "name": "occluded"},
]

SPLIT_MAP = {"train": "train", "val": "valid"}


def main():
    for split, out_name in SPLIT_MAP.items():
        images_list = []
        annotations_list = []
        img_id = 0
        ann_id = 0

        out_split_dir = OUT_DIR / out_name
        out_split_dir.mkdir(parents=True, exist_ok=True)

        for class_name, class_id in CLASS_MAP.items():
            src_dir = BAKALAURS_DIR / split / class_name

            for img_path in sorted(src_dir.glob("*.jpg")):
                json_path = img_path.with_suffix(".json")

                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)

                img_w = data["width"]
                img_h = data["height"]

                images_list.append({
                    "id":        img_id,
                    "file_name": img_path.name,
                    "width":     img_w,
                    "height":    img_h,
                })

                for obj in data["objects"]:
                    b = obj["bbox"]
                    x = b["xmin"]
                    y = b["ymin"]
                    w = b["xmax"] - b["xmin"]
                    h = b["ymax"] - b["ymin"]
                    annotations_list.append({
                        "id":          ann_id,
                        "image_id":    img_id,
                        "category_id": class_id,
                        "bbox":        [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                        "area":        round(w * h, 2),
                        "iscrowd":     0,
                    })
                    ann_id += 1

                dst = out_split_dir / img_path.name
                if not dst.exists():
                    rel = os.path.relpath(img_path.resolve(), dst.parent)
                    os.symlink(rel, dst)

                img_id += 1

        coco = {
            "images":      images_list,
            "annotations": annotations_list,
            "categories":  CATEGORIES,
        }
        with open(out_split_dir / "_annotations.coco.json", "w", encoding="utf-8") as f:
            json.dump(coco, f)


if __name__ == "__main__":
    main()
