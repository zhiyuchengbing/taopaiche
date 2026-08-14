# -*- coding: utf-8 -*-
"""从 split_v2/runs/truck_train08134 的 last.pt 续训（中断恢复）。

后台任务中断后 best.pt 与 last.pt 均完整，续训不丢 best=0.99496 基准，
从 last.pt(epoch 27) 继续至 epochs=200。resume=True 会自动读取原 args.yaml。
"""
from ultralytics import YOLO

LAST = r"D:\data2\truck\split_v2\runs\truck_train08134\weights\last.pt"


def main() -> int:
    model = YOLO(LAST)
    model.train(
        resume=True,
        device="0",
        workers=2,  # 无页面文件，保持 workers=2 防 WinError1455
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
