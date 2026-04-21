# Decision Rules

## Stable labels

Use only these canonical labels unless the whole call chain is updated:

- `fake_plate`
- `change_trailer`
- `normal`
- `abnormal`

## Evidence hierarchy

Prefer structural evidence over lighting evidence.

Strong evidence:

- front grille shape
- bumper openings
- headlight contour
- mirror position
- trailer rear geometry
- tail-light layout
- reflective strip layout

Weak evidence:

- brightness
- shadow
- headlight on/off
- tail-light on/off
- camera angle differences that do not change structure

## Practical repository guidance

- If the same plate appears with clearly different front structure, treat it as `fake_plate`.
- If the tractor head remains consistent but the trailer rear differs, treat it as `change_trailer`.
- If the apparent difference is mostly caused by night lighting or exposure changes, keep it `normal`.
- If parsing, model loading, or preprocessing fails, preserve `abnormal` behavior instead of fabricating a normal label.

## Prompt design guidance

- Ask the vision model to focus on stable structure, not illumination.
- Keep the final line easy to parse.
- Restrict the final line to the allowed labels for that subtask.
- If the model returns extra explanation, parse from the last line first and fall back to keyword matching.
