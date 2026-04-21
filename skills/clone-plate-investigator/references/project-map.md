# Project Map

## Main files

- `d:\project\README.md`
  - Project history, threshold notes, API contracts, case type meanings, and feature timeline.
- `d:\project\data_chuli\demo\demo\my_predict_gui_new.py`
  - Main Flask service and current production-style decision flow.
- `d:\project\data_chuli\demo\demo\Siamese-pytorch-master\qwen_vl\predict_ai.py`
  - AI vision checker and prompt design for full vehicle, head, tail, and difference analysis.
- `d:\project\data_chuli\demo\demo\Siamese-pytorch-master\detect_clone_plates.py`
  - Batch CSV-based clone-plate detection workflow.
- `d:\project\data_chuli\cropper.py`
  - Vehicle crop preprocessing entry point.

## Typical workflow in this repository

1. Load or receive two source vehicle images.
2. Crop the vehicle area first.
3. Crop head and tail parts from the processed vehicle image.
4. Run Siamese similarity on head and tail.
5. Map probabilities into a first-pass `case_type`.
6. For abnormal cases, optionally run AI recheck or fine-grained difference analysis.
7. Return or persist `case_type`, probabilities, previews, and optional review fields.

## Where to inspect based on task type

- Need to change prompts or result extraction:
  - inspect `qwen_vl/predict_ai.py`
- Need to change thresholds or verdict combination:
  - inspect `my_predict_gui_new.py`
- Need to change batch offline checking:
  - inspect `detect_clone_plates.py`
- Need to change vehicle crop preprocessing:
  - inspect `data_chuli/cropper.py`
- Need to understand business expectations:
  - inspect `README.md`
