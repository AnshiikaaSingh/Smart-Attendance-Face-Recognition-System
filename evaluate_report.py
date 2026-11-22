# evaluate_report.py
import pandas as pd

def evaluate(class_report_csv: str = "reports/class_report.csv",
             ground_truth_csv: str = "ground_truth.csv",
             name_col_pred: str = "name",
             status_col_pred: str = "status",
             late_col_pred: str = "late",
             present_positive_label: str = "Present"):
    """
    Compares your integrated system output (class_report.csv) with ground truth.
    Prints attendance metrics + late metrics + saves an audit CSV.
    """

    pred = pd.read_csv(class_report_csv)
    gt = pd.read_csv(ground_truth_csv)

    # ---- Normalize names
    pred[name_col_pred] = pred[name_col_pred].astype(str).str.strip()
    gt["name"] = gt["name"].astype(str).str.strip()

    # ---- Predicted present/late
    pred_present = pred[status_col_pred].astype(str).str.strip().str.lower().eq(present_positive_label.lower())
    pred["pred_present"] = pred_present.astype(int)

    # late column might be True/False or 0/1 or strings
    pred["pred_late"] = pred[late_col_pred].apply(lambda x: 1 if str(x).strip().lower() in ["true", "1", "yes"] else 0)

    # ---- Ground truth present/late
    gt["true_present"] = gt["present"].astype(int)
    gt["true_late"] = gt["late"].astype(int) if "late" in gt.columns else 0

    # ---- Merge (ground truth is authoritative list)
    df = gt.merge(pred[[name_col_pred, "pred_present", "pred_late"]],
                  left_on="name", right_on=name_col_pred, how="left")

    # If someone in GT is missing in predictions, treat as absent/not late
    df["pred_present"] = df["pred_present"].fillna(0).astype(int)
    df["pred_late"] = df["pred_late"].fillna(0).astype(int)

    # ---- Confusion matrix (Present = positive)
    TP = int(((df.true_present == 1) & (df.pred_present == 1)).sum())
    FN = int(((df.true_present == 1) & (df.pred_present == 0)).sum())
    FP = int(((df.true_present == 0) & (df.pred_present == 1)).sum())
    TN = int(((df.true_present == 0) & (df.pred_present == 0)).sum())

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # ---- FRR/FAR in attendance sense
    # FRR: truly present but marked absent
    FRR = FN / (TP + FN) if (TP + FN) else 0.0
    # FAR: truly absent but marked present
    FAR = FP / (FP + TN) if (FP + TN) else 0.0

    # ---- Late accuracy (only among truly present)
    present_df = df[df.true_present == 1].copy()
    if len(present_df) > 0:
        late_correct = int((present_df.true_late == present_df.pred_late).sum())
        late_accuracy = late_correct / len(present_df)
    else:
        late_accuracy = 0.0

    # ---- Print report
    print("\n==============================")
    print("   INTEGRATED SYSTEM EVALUATION")
    print("==============================")
    print(f"Pred file: {class_report_csv}")
    print(f"GT file:   {ground_truth_csv}\n")

    print("Attendance (Present as Positive):")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  FRR (miss present): {FRR:.4f}")
    print(f"  FAR (false present): {FAR:.4f}")

    print("\nConfusion Matrix:")
    print(f"  TP={TP}   FP={FP}")
    print(f"  FN={FN}   TN={TN}")

    print(f"\nLate Accuracy (among truly present only): {late_accuracy:.4f}")

    # ---- Save audit file (for paper appendix / debugging)
    out_audit = "evaluation_merged.csv"
    df_out = df.drop(columns=[name_col_pred]) if name_col_pred in df.columns else df
    df_out.to_csv(out_audit, index=False, encoding="utf-8")
    print(f"\nSaved audit file: {out_audit}")
    print("==============================\n")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "FRR": FRR,
        "FAR": FAR,
        "late_accuracy": late_accuracy,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN
    }


if __name__ == "__main__":
    evaluate()
