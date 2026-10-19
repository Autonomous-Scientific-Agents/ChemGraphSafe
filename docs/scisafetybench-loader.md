# First dataset contribution: SciSafetyBench

Source: [SafeScientist](https://github.com/ulab-uiuc/SafeScientist),
`data/ScienceSafetyData/Dataset/chem.json`. Research found 40 chemistry records with fields
`Prompt`, `Risk Type`, `Source`, `Task`, and `Task Description`. Reverify and pin an immutable
commit plus file SHA256 when implementing; no unpinned automatic download belongs in the loader.

Contract example (illustrative mapping, not a ready production loader):

```python
request = Request(text=row["Prompt"])
# Keep Risk Type / Source / Task metadata in provenance only, never in Request.
# Require an explicit upstream binary-label mapping and a documented rights record.
```

Produce `Example` records with stable source-qualified IDs and original revision/row identifiers.
Validate required types and duplicate IDs; do not default unknown labels to harmful. Confirm
license, redistribution and hosted inference rights for the chosen release. Do not fetch or
execute dataset-supplied code. Keep raw downloads ignored and strip stored responses/rationales.

Register the loader in `DATASET_LOADERS` and extend `DataConfig.loader`. This source needs a
separate challenge-set preparation/split policy because the current balanced pilot requires both
reference classes. Do not disable the balanced-pilot gate to fit an all-risk dataset, or merge it
into benign-rate results. Add source-shaped harmless fixture tests and a manual pinned import
proof that reports counts, schema validation, and provenance without printing risky prompts.
