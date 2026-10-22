"""Source-category classification metrics and rendering, separate from triage semantics."""

from .models import RISK_TYPES, Prediction, policy_labels
from .reporting import _fscore, rate
from .scisafetybench import DATA_PATH, RECORDS_PER_FILE, REVISION, SOURCE_FILES, example_id
from .storage import atomic_text, digest


def validate_references(refs: list[dict], *, mock: bool):
    for ref in refs:
        if ref.get("annotation") is not None:
            raise ValueError("Risk-type reports cannot use annotations")
        label, origin = ref.get("source_risk_type"), ref.get("reference_origin")
        if ref["split"] == "smoke":
            if label is not None or origin is not None:
                raise ValueError("Risk-type smoke must not have classification references")
            continue
        if ref["split"] != "eval" or label not in RISK_TYPES:
            raise ValueError("Risk-type evaluation requires a source category for every case")
        if ref.get("source_label") is not None or ref["category_id"] != digest(label)[:12]:
            raise ValueError("Risk-type reference category differs from its source metadata")
        if ref["source"] != "SciSafetyBench":
            if not mock or origin != "owned_fixture":
                raise ValueError(
                    "Only offline fixtures may substitute for source-native references"
                )
            continue
        source_id = ref.get("source_id", "")
        filename, _, index = source_id.partition(":")
        if (
            origin != "source_native"
            or ref.get("source_revision") != REVISION
            or filename not in SOURCE_FILES
            or not index.isdecimal()
            or not 0 <= int(index) < RECORDS_PER_FILE
            or ref.get("source_split") != f"{DATA_PATH}/{filename}"
            or ref["domain"] != SOURCE_FILES[filename]["domain"]
            or ref["id"] != example_id(filename, int(index))
        ):
            raise ValueError("Invalid source-native risk-type reference provenance")


def risk_metrics(refs: list[dict], predictions: dict[str, Prediction], *, policy: dict) -> dict:
    labels = policy_labels(policy)
    pairs = [
        (
            ref,
            predictions.get(ref["id"], Prediction(status="error", error_code="missing_prediction")),
        )
        for ref in refs
    ]
    for _, prediction in pairs:
        prediction.validate_policy(policy)
    scored = [(ref, pred) for ref, pred in pairs if ref.get("source_risk_type") is not None]
    confusion = {label: dict.fromkeys((*labels, "ABSTAIN", "ERROR"), 0) for label in labels}
    for ref, pred in scored:
        truth = ref["source_risk_type"]
        if truth not in labels:
            raise ValueError("Reference category is outside the active risk-type labels")
        confusion[truth][pred.decision if pred.status == "ok" else pred.status.upper()] += 1
    precision, recall, f1s, counts = {}, {}, {}, {}
    for label in labels:
        support = counts[label] = sum(confusion[label].values())
        correct = confusion[label][label]
        predicted = sum(confusion[truth][label] for truth in labels)
        precision[label] = rate(correct, predicted)
        recall[label] = rate(correct, support)
        f1s[label] = _fscore(correct, predicted - correct, support - correct) if support else None
    origins = sorted({ref["reference_origin"] for ref, _ in scored})
    return {
        "n": len(refs),
        "completed": sum(pred.status != "error" for _, pred in pairs),
        "operationally_complete": all(pred.status != "error" for _, pred in pairs),
        "decision_coverage": rate(sum(pred.status == "ok" for _, pred in pairs), len(refs)),
        "abstention_overall": rate(sum(pred.status == "abstain" for _, pred in pairs), len(refs)),
        "error_rate": rate(sum(pred.status == "error" for _, pred in pairs), len(refs)),
        "reference": origins[0] if len(origins) == 1 else "mixed" if origins else None,
        "reference_coverage": rate(len(scored), len(refs)),
        "reference_counts": counts,
        "accuracy": rate(sum(confusion[label][label] for label in labels), len(scored)),
        "confusion": confusion,
        "precision": precision,
        "recall": recall,
        "f1": f1s,
        "macro_f1": sum(f1s.values()) / len(labels)
        if all(value is not None for value in f1s.values())
        else None,
    }


def render_risk_report(run, manifest, result, comparison, error_count):
    def pct(value):
        return "N/A" if value is None else f"{value:.1%}"

    title = "SciSafetyBench risk-type classification"
    if manifest["mock"]:
        title += " — MOCK / OFFLINE demonstration"
    text = [
        f"# {title}",
        "",
        "Scores measure agreement with source risk categories, not the original benchmark's "
        "agent-safety evaluation. Owned fixture results are explicitly marked as mock.",
        "",
        "| Model | Reference | N | Complete | Accuracy | Macro-F1 | Decision coverage |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in comparison:
        text.append(
            f"| {row['model']} | {row['reference'] or 'unscored smoke'} | {row['n']} | "
            f"{row['complete']} | {pct(row['accuracy'])} | {pct(row['macro_f1'])} | "
            f"{pct(row['decision_coverage'])} |"
        )
    outcomes = (*RISK_TYPES, "ABSTAIN", "ERROR")
    for spec in manifest["models"]:
        summary = result["models"][spec["name"]]
        primary = summary["splits"].get("eval", summary["splits"].get("smoke"))
        text += ["", f"## {spec['name']}", ""]
        if not primary["reference_coverage"]["count"]:
            text += ["Transport/schema smoke only; no classification references or accuracy."]
            continue
        text += [
            "| Reference | " + " | ".join(outcomes) + " |",
            "|---|" + "---:|" * len(outcomes),
        ]
        for label, counts in primary["confusion"].items():
            text.append(f"| {label} | " + " | ".join(str(counts[k]) for k in outcomes) + " |")
        text += [
            "",
            "| Category | Support | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|",
        ]
        for label in RISK_TYPES:
            text.append(
                f"| {label} | {primary['reference_counts'][label]} | "
                f"{pct(primary['precision'][label]['value'])} | "
                f"{pct(primary['recall'][label]['value'])} | {pct(primary['f1'][label])} |"
            )
        text += ["", "| Domain | N | Accuracy | Macro-F1 |", "|---|---:|---:|---:|"]
        for domain, scores in summary["slices"].get("domain", {}).items():
            text.append(
                f"| {domain} | {scores['n']} | {pct(scores['accuracy']['value'])} | "
                f"{pct(scores['macro_f1'])} |"
            )
    text += [
        "",
        f"Disagreement/operational queue: {error_count} ID-only records in errors.jsonl "
        "and errors.csv.",
        "See metrics.json for class/domain slices, denominators, Wilson intervals, model "
        "identities, latency, attempts, token usage and costs. "
        "Constant-category baselines are diagnostic only.",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in result["limitations"]],
    ]
    if any(summary["mixed_identities"] for summary in result["models"].values()):
        text.append(
            "- MIXED RETURNED MODEL IDENTITIES: inspect per_identity; no stable-checkpoint claim."
        )
    atomic_text(run / "report.md", "\n".join(text) + "\n")
    return {
        "report": str(run / "report.md"),
        "mock": manifest["mock"],
        "models": len(manifest["models"]),
        "errors": error_count,
    }
