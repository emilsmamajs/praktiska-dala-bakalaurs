import os
from pathlib import Path
from ultralytics import YOLO

SCRIPT_DIR = Path(os.environ.get("YOLO_WORKDIR", Path(__file__).resolve().parent))
DATA_YAML  = SCRIPT_DIR.parent / "yolo_dataset" / "data.yaml"


def main() -> None:
    model = YOLO("yolo26l.pt")

    train_args: dict = dict(
        data        = str(DATA_YAML),
        epochs      = 150,
        batch       = 4,
        imgsz       = 1280,
        workers     = 0,
        project     = str(SCRIPT_DIR / "runs"),
        patience    = 20,
        save        = True,
        plots       = True,
        verbose     = True,
    )

    model.train(**train_args)

    metrics = model.val()


if __name__ == "__main__":
    main()
