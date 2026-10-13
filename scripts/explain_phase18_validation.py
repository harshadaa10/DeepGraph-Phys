from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "features" / "phase11d_expanded_features.csv"
BUNDLE_PATH = ROOT / "models" / "phase16_baseline_60.joblib"
OUTPUT_DIR = ROOT / "outputs" / "experiments"
FIGURES_DIR = ROOT / "outputs" / "figures"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(BUNDLE_PATH)
    feature_columns = bundle["feature_columns"]
    scaler = bundle["scaler"]
    model = bundle["model"]
    threshold = float(bundle["threshold"])

    # Retain validation rows only. Do not evaluate the reserved test split.
    validation_parts = []
    for chunk in pd.read_csv(FEATURES_PATH, chunksize=25):
        validation_parts.append(
            chunk.loc[chunk["split"].eq("validation")].copy()
        )

    validation_df = pd.concat(validation_parts, ignore_index=True)
    if validation_df.empty:
        raise ValueError("No validation rows were found.")

    if len(validation_df) != 16:
        raise ValueError(
            f"Expected 16 validation videos, found {len(validation_df)}."
        )

    missing = [col for col in feature_columns if col not in validation_df.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    X = validation_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad_columns = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Missing/non-numeric feature values: {bad_columns}")

    # Transform using the saved scaler. Never fit or update it here.
    X_scaled = scaler.transform(X.to_numpy(dtype=float))
    scores = -model.decision_function(X_scaled)
    predictions = np.where(scores >= threshold, "fake", "real")

    summary = pd.DataFrame({
        "video": validation_df["video"].astype(str),
        "true_label": validation_df["label"].astype(str).str.lower(),
        "anomaly_score": scores,
        "threshold": threshold,
        "predicted_label": predictions,
    })
    summary["correct"] = summary["true_label"].eq(summary["predicted_label"])
    summary_path = OUTPUT_DIR / "phase18_validation_predictions.csv"
    summary.to_csv(summary_path, index=False)

    # Standardized deviations relative to the genuine-training reference.
    deviations = pd.DataFrame(
        X_scaled, columns=feature_columns, index=validation_df.index
    )

    explanation_rows = []
    for row_idx in range(len(validation_df)):
        order = np.argsort(np.abs(deviations.iloc[row_idx].to_numpy()))[::-1]
        for rank, feature_idx in enumerate(order, start=1):
            explanation_rows.append({
                "video": str(validation_df.iloc[row_idx]["video"]),
                "true_label": str(validation_df.iloc[row_idx]["label"]).lower(),
                "predicted_label": str(predictions[row_idx]),
                "anomaly_score": float(scores[row_idx]),
                "threshold": threshold,
                "feature": feature_columns[feature_idx],
                "standardized_value": float(deviations.iloc[row_idx, feature_idx]),
                "absolute_deviation": float(
                    abs(deviations.iloc[row_idx, feature_idx])
                ),
                "deviation_rank": rank,
            })

    explanation_df = pd.DataFrame(explanation_rows)
    explanation_path = OUTPUT_DIR / "phase18_validation_feature_deviations.csv"
    explanation_df.to_csv(explanation_path, index=False)

    # Aggregate descriptive visualization: no feature-importance claims.
    mean_deviation = deviations.abs().mean().sort_values(ascending=False)
    top_features = mean_deviation.head(15).sort_values()

    plt.figure(figsize=(11, 7))
    top_features.plot(kind="barh")
    plt.xlabel("Mean absolute standardized deviation")
    plt.ylabel("Feature")
    plt.title("Phase 18: Most unusual features across validation videos")
    plt.tight_layout()

    figure_path = FIGURES_DIR / "phase18_validation_feature_deviations.png"
    plt.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("PHASE 18 — VALIDATION EXPLANATIONS CREATED")
    print(f"Validation videos: {len(validation_df)}")
    print(f"Features analyzed per video: {len(feature_columns)}")
    print(f"Frozen threshold: {threshold:.8f}")
    print(f"Prediction summary: {summary_path}")
    print(f"Feature deviations: {explanation_path}")
    print(f"Visualization: {figure_path}")
    print("Reserved test split: not evaluated")
    print("Note: deviations are descriptive, not causal feature attributions.")


if __name__ == "__main__":
    main()

