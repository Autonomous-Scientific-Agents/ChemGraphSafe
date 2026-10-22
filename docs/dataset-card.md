# Dataset card: provisional SciHazard chemistry pilot

Source: [AI45Lab/DeepSafe-Sci](https://github.com/AI45Lab/DeepSafe-Sci), commit
`f8809e5af23495e469dfc53e99973f74031fcb11`. See the upstream
[dataset card](https://github.com/AI45Lab/DeepSafe-Sci/blob/f8809e5af23495e469dfc53e99973f74031fcb11/DATASET_CARD.md)
and [data license](https://github.com/AI45Lab/DeepSafe-Sci/blob/f8809e5af23495e469dfc53e99973f74031fcb11/LICENSE-DATA).
Attribution belongs to the DeepSafe-Sci/SciHazard contributors, AI45Lab. Data are CC BY 4.0;
the project's Apache-2.0 code license does not replace their terms.

The importer reads `DeHarmScore-trace/dataset/600safe_final.jsonl` and
`2400unsafe_final.jsonl`. SHA256 values are pinned in `data.py` and copied to preparation
manifests. Source files remain in ignored `.local/data/pilot-v1/raw/`.

## Transformations and counts

The pinned files contain 50 safe-split and 200 harmful-split Chemistry records. Of the latter,
22 also carry `is_harmful=false`. Their intent is unresolved: the importer preserves the split
label and native flag separately and quarantines these conflicts without relabeling them.
`feasibility.json` records their source-qualified IDs. The 228 eligible rows have no exact
duplicates and form 228 singleton groups at the fixed similarity threshold.

Normalization uses Unicode NFC and whitespace, preserving case. Exact duplicates with
conflicting reference labels are quarantined; same-label duplicates preserve related IDs.
Character 3–5-gram TF-IDF with cosine >=0.9 and explicit variant-family links define groups
before sampling. This is a conservative heuristic, not proof of semantic independence.
Category and subject are never treated as variant families.

The deterministic seed-42 pilot selects one case from each of 100 groups: 50 benign and
50 harmful, then stratifies into 20 development and 80 evaluation cases (10/10 and 40/40).
All 50 eligible benign examples are used, so benign coverage is narrow. The primary reference
is the original release split, not independently audited policy truth.

Only the original question enters `Request.text`; `context` is empty for this release.
Source responses, reasoning, checklists, and risk metadata are excluded from classifier input.
Annotations, when later created, live in a separate file and remain explicitly provisional.
Current delivery freezes no three-way policy labels. Aggregate reports contain ID-only errors;
join to local examples for inspection. Do not redistribute raw rows through issues or pull requests.

## Limits and other sources

This small English pilot measures request triage and source-label agreement. It does not measure
chemical knowledge, agent behavior, response safety, or deployment reliability. Upstream label
errors, model training overlap, subtle duplicates, and sampling bias remain possible.

SciSafetyBench is implemented as a separate 80-case chemistry/materials challenge with one shared
oversight policy and provisional rule-derived references; see its [dataset and loader contract](scisafetybench-loader.md).
SOSBench and SafeSci are opt-in future imports with source-specific access and processing checks.
WMDP-Chem's knowledge questions and GHS molecular hazard labels must not be converted directly
into this request-triage task. The original ChemSafetyBench prompt benchmark remains deferred
until an accessible release is verified.
