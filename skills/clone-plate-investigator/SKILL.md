---
name: clone-plate-investigator
description: Analyze suspected clone-plate vehicle cases in this repository by reusing the existing vehicle cropper, Siamese comparison, head and tail part analysis, and AI recheck pipeline. Use when Codex needs to identify whether two vehicle images are a fake_plate, change_trailer, or normal case; trace the current decision logic in this project; adjust thresholds or prompts for clone-plate recognition; or debug and extend clone-plate detection workflows in this codebase.
---

# Clone Plate Investigator

Use this skill when working on the clone-plate recognition pipeline in this repository.

## Follow the repository-first workflow

1. Start from the current repository instead of inventing a new algorithm.
2. Reuse the existing path for:
   - vehicle crop and preprocessing
   - head and tail crop
   - Siamese similarity
   - AI recheck and difference analysis
3. Keep naming aligned with the existing business labels:
   - `fake_plate`
   - `change_trailer`
   - `normal`
   - `abnormal`

## Read the local references before changing logic

- Read `references/project-map.md` to find the main entry points in this repo.
- Read `references/decision-rules.md` when the task touches thresholds, verdict mapping, or review language.

## Prefer these implementation rules

1. Treat lighting, shadow, reflection, and vehicle lamps as weak evidence.
2. Treat stable structure as strong evidence:
   - front grille
   - headlights outline
   - bumper structure
   - mirrors
   - trailer rear layout
   - tail-light arrangement
3. If the head structure is clearly inconsistent under the same plate, prefer `fake_plate`.
4. If the tractor head is consistent but the trailer rear structure differs, prefer `change_trailer`.
5. If differences are mostly caused by illumination or exposure, prefer `normal`.
6. Preserve existing threshold semantics unless the user explicitly asks to change them.

## Preferred repo entry points

- Main service and end-to-end business flow:
  - `data_chuli/demo/demo/my_predict_gui_new.py`
- Vision-model prompt logic:
  - `data_chuli/demo/demo/Siamese-pytorch-master/qwen_vl/predict_ai.py`
- Batch clone-plate checking:
  - `data_chuli/demo/demo/Siamese-pytorch-master/detect_clone_plates.py`
- Vehicle cropper:
  - `data_chuli/cropper.py`

## When adding or modifying clone-plate capability

1. Confirm whether the change belongs in:
   - preprocessing
   - Siamese thresholding
   - AI recheck prompt design
   - result normalization
   - UI/API output
2. Patch the narrowest layer that solves the user request.
3. Keep result words machine-stable. Avoid introducing new case labels unless the caller is updated too.
4. If prompts are changed, keep the final output line easy to parse and limited to known labels.
5. If logic is changed, inspect downstream places that persist or display:
   - `case_type`
   - `diff_desc`
   - review/export fields

## Validation checklist

1. Verify the final labels remain `fake_plate`, `change_trailer`, `normal`, or `abnormal`.
2. Verify prompt parsing still works when the model outputs extra explanation.
3. Verify downstream callers still understand the returned fields.
4. If the change affects thresholds, confirm the README or local docs are still consistent.
5. If tests are absent, at least inspect the main call chain and explain residual risk.
