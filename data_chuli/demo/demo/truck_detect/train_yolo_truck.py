import os
from ultralytics import YOLO


DATA_YAML = r"D:\data2\truck\split\data.yaml"  # TODO: 改成你自己的yaml路径
WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "yolov8s.pt")

EPOCHS = 200
IMGSZ = 640
BATCH = 16
DEVICE = "0"
PROJECT_DIR = r"D:\data2\truck\split\runs"
RUN_NAME = "truck_train0321"


def main() -> int:
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"dataset yaml not found: {DATA_YAML}")

    weights = WEIGHTS if os.path.exists(WEIGHTS) else "D:\project\data_chuli\demo\yolov8s.pt"

    model = YOLO(weights)
    model.train(
        data=DATA_YAML,
        epochs=int(EPOCHS),
        imgsz=int(IMGSZ),
        batch=int(BATCH),
        device=str(DEVICE),
        project=str(PROJECT_DIR),
        name=str(RUN_NAME),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
