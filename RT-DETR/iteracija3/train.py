import os
from pathlib import Path
from rfdetr import RFDETRLarge

SCRIPT_DIR  = Path(os.environ.get("RFDETR_WORKDIR", Path(__file__).resolve().parent))
DATASET_DIR = SCRIPT_DIR.parent / "coco_dataset"


def main() -> None:
    model = RFDETRLarge(resolution=640)
    output_dir = str(SCRIPT_DIR / "runs" / "train")

    model.train(
        dataset_dir              = str(DATASET_DIR),
        epochs                   = 150,
        batch_size               = 4,
        grad_accum_steps         = 4,
        lr                       = 5e-5,
        output_dir               = output_dir,
        early_stopping           = True,
        early_stopping_patience  = 20,
        early_stopping_min_delta = 0.001,
        early_stopping_use_ema   = True,
    )


if __name__ == "__main__":
    main()
