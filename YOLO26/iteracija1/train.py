import os
from pathlib import Path
from ultralytics import YOLO

SCRIPT_DIR = Path(os.environ.get("YOLO_WORKDIR", Path(__file__).resolve().parent))
DATA_YAML  = SCRIPT_DIR.parent / "yolo_dataset" / "data.yaml"


def main() -> None:
    model = YOLO("yolo26m.pt")

    train_args: dict = dict(
        data        = str(DATA_YAML),
        epochs      = 100,
        batch       = 16,
        imgsz       = 640,
        workers     = 8,
        project     = str(SCRIPT_DIR / "runs"),
        patience    = 15,
        save        = True,
        plots       = True,
        verbose     = True,
    )

    model.train(**train_args)

    metrics = model.val()


if __name__ == "__main__":
    main()
