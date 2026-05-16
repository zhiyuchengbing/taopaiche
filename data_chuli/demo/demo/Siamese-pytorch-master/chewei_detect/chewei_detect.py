from pathlib import Path
from typing import Union

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_WEIGHTS = Path(r"D:\project\data_chuli\demo\demo\Siamese-pytorch-master\chewei_detect\weight\best.pt")
DEFAULT_TEST_IMAGE = Path(
    r"D:\data2\weibu_data\\01_01_selected_17_2_20260101_005250_CH01_0002_172601010001_000001.jpg"
)
DEFAULT_OUTPUT = Path(r"D:\data2\\crop_vehicle_yolo_result.jpg")
VEHICLE_CLASSES = {"truck"}


def imread_unicode(image_path: Path):
    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(save_path: Path, image) -> bool:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ext = save_path.suffix or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(str(save_path))
    return True


class VehicleCropper:
    def __init__(
        self,
        weights_path: Union[str, Path] = DEFAULT_WEIGHTS,
        conf: float = 0.25,
        imgsz: int = 640,
    ):
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")

        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLO(str(self.weights_path))

    def _get_class_name(self, class_id: int) -> str:
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _select_largest_vehicle_box(self, result, image_shape):
        if result.boxes is None or len(result.boxes) == 0:
            return None

        height, width = image_shape[:2]
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        candidates = []
        for box, class_id in zip(boxes, classes):
            class_name = self._get_class_name(class_id).lower()
            if class_name not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box
            xmin = max(0, min(int(round(x1)), width - 1))
            ymin = max(0, min(int(round(y1)), height - 1))
            xmax = max(xmin + 1, min(int(round(x2)), width))
            ymax = max(ymin + 1, min(int(round(y2)), height))
            area = max(0, xmax - xmin) * max(0, ymax - ymin)

            candidates.append((area, xmin, ymin, xmax, ymax, class_name))

        if not candidates:
            return None

        _, xmin, ymin, xmax, ymax, class_name = max(candidates, key=lambda item: item[0])
        return {
            "name": class_name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }

    def crop_image(self, image_path: Union[str, Path]):
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = imread_unicode(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        results = self.model.predict(source=image, conf=self.conf, imgsz=self.imgsz, verbose=False)
        if not results:
            return image, None

        box = self._select_largest_vehicle_box(results[0], image.shape)
        if box is None:
            return image, None

        crop = image[box["ymin"] : box["ymax"], box["xmin"] : box["xmax"]]
        if crop.size == 0:
            return image, None

        return crop, box

    def crop_and_save(self, image_path: Union[str, Path], output_path: Union[str, Path]):
        cropped, box = self.crop_image(image_path)
        output_path = Path(output_path)
        if not imwrite_unicode(output_path, cropped):
            raise RuntimeError(f"Failed to save image: {output_path}")
        return output_path, box


def main():
    cropper = VehicleCropper()
    output_path, box = cropper.crop_and_save(DEFAULT_TEST_IMAGE, DEFAULT_OUTPUT)

    print(f"Image: {DEFAULT_TEST_IMAGE}")
    print(f"Saved: {output_path}")
    if box is None:
        print("Detection: none, saved original image")
    else:
        print(
            f"Detection: {box['name']} "
            f"({box['xmin']}, {box['ymin']}, {box['xmax']}, {box['ymax']})"
        )


if __name__ == "__main__":
    main()
