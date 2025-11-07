import json
import sys

# Usage:
#   python eval_harness.py /path/to/ground_truth.jsonl /path/to/model_outputs.jsonl
#
# ground_truth.jsonl lines like:
#   {"id":"123", "ground_truth": {"sex":"male","age":54,"systolic_bp":120,"diastolic_bp":80,"diagnosis":"..."}}
#
# model_outputs.jsonl lines like:
#   {"id":"123", "pred": {"sex":"M","age":54,"systolic_bp":118,"diastolic_bp":80,"diagnosis":"..."}}  # prediction format

NORM_SEX = {"m":"male","male":"male","f":"female","female":"female","xy":"male","xx":"female"}

FIELDS = ["sex","age","systolic_bp","diastolic_bp","heart_rate","diagnosis","treatment","outcome"]

def norm(v, field):
    if v is None:
        return None
    if field in ("systolic_bp","diastolic_bp","age","heart_rate"):
        try:
            return int(float(v))
        except Exception:
            return None
    if field == "sex":
        s = str(v).strip().lower()
        return NORM_SEX.get(s, s)
    # strings
    return str(v).strip().lower()

def bp_close(a, b, tol=5):
    if a is None or b is None:
        return False
    return abs(a - b) <= tol

def load_jsonl(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                data[str(obj["id"])] = obj
    return data

def macro_f1_from_counts(tp, fp, fn):
    # avoid div by zero
    precision = tp / (tp + fp) if (tp+fp) else 0.0
    recall = tp / (tp + fn) if (tp+fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def main(gt_path, pred_path):
    gt_map = load_jsonl(gt_path)
    pr_map = load_jsonl(pred_path)

    per_field_correct = {f:0 for f in FIELDS}
    per_field_total = {f:0 for f in FIELDS}

    # For a crude F1, treat each exact-field-match as a "label"
    tp = fp = fn = 0

    for rid, g in gt_map.items():
        gt = g.get("ground_truth", {})
        pred = pr_map.get(rid, {}).get("pred", {})
        for field in FIELDS:
            gtv = norm(gt.get(field), field)
            prv = norm(pred.get(field), field)
            if gtv is None:
                continue
            per_field_total[field] += 1

            correct = False
            if field in ("systolic_bp","diastolic_bp"):
                correct = bp_close(gtv, prv, tol=5)
            else:
                correct = (gtv == prv)

            if correct:
                per_field_correct[field] += 1
                tp += 1
            else:
                fn += 1
                if prv is not None:
                    fp += 1

    # Accuracy: average over fields present
    totals = sum(per_field_total.values())
    corrects = sum(per_field_correct.values())
    accuracy = corrects / totals if totals else 0.0
    macro_f1 = macro_f1_from_counts(tp, fp, fn)

    report_lines = []
    for f in FIELDS:
        if per_field_total[f]:
            accf = per_field_correct[f] / per_field_total[f]
            report_lines.append(f"{f:14s} acc={accf:.3f} ({per_field_correct[f]}/{per_field_total[f]})")

    out = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "field_breakdown": report_lines
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python eval_harness.py synthetic_notes.jsonl model_outputs.jsonl")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
