from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Tuple
from app.config import settings


class SHAPService:
    @staticmethod
    def get_explainer_and_values(
        predictor: Any, X_preprocessed: np.ndarray
    ) -> Tuple[Any, np.ndarray]:
        """
        Initializes the appropriate SHAP explainer and computes SHAP values.
        Handles TreeExplainer fallback to LinearExplainer/Explainer.
        """
        # Choose explainer
        class_name = predictor.__class__.__name__
        is_tree = any(x in class_name.lower() for x in ["forest", "tree", "xgb", "lgb", "cat"])
        
        # We sample background data if it's large to keep it fast
        background = X_preprocessed
        if background.shape[0] > 100:
            # Random sample
            indices = np.random.choice(background.shape[0], 100, replace=False)
            background = background[indices]

        try:
            if is_tree:
                explainer = shap.TreeExplainer(predictor)
                # CatBoost and XGBoost sometimes need specific handling
                shap_values = explainer.shap_values(X_preprocessed)
            else:
                explainer = shap.LinearExplainer(predictor, background)
                shap_values = explainer.shap_values(X_preprocessed)
        except Exception:
            # Generic fallback explainer
            try:
                explainer = shap.Explainer(predictor, background)
                shap_values = explainer(X_preprocessed).values
            except Exception:
                # Absolute fallback: kernel explainer or random values if it fails completely
                # (to ensure application never crashes in demo)
                explainer = shap.KernelExplainer(predictor.predict, background[:10])
                shap_values = explainer.shap_values(X_preprocessed)

        # Handle different SHAP shapes (multiclass lists, binary shapes)
        if isinstance(shap_values, list):
            # For random forest classification, list of arrays [class_0, class_1]
            if len(shap_values) == 2:
                # Binary: take positive class contributions
                shap_values = shap_values[1]
            else:
                # Multiclass: default to taking first class, or let's sum absolute contributions
                shap_values = shap_values[0]
        elif len(shap_values.shape) == 3:
            # New SHAP API returns shape (n_samples, n_features, n_classes)
            # Take positive class index 1 if binary, or index 0
            if shap_values.shape[2] == 2:
                shap_values = shap_values[:, :, 1]
            else:
                shap_values = shap_values[:, :, 0]

        return explainer, shap_values

    @staticmethod
    def generate_global_importance(
        dataset_id: int,
        model_payload: Dict[str, Any],
        df_cleaned: pd.DataFrame,
    ) -> str:
        """
        Computes global SHAP values, generates a summary plot, and saves it.
        Returns the absolute storage path.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pipeline = model_payload["pipeline"]
        feature_cols = model_payload["feature_cols"]
        target_col = model_payload["target_col"]

        preprocessor = pipeline.named_steps["preprocessor"]
        predictor = pipeline.named_steps["predictor"]

        # Drop target if present
        X = df_cleaned[feature_cols]
        X_preprocessed = preprocessor.transform(X)

        # Handle sparse matrix output from OneHotEncoder
        if hasattr(X_preprocessed, "toarray"):
            X_preprocessed = X_preprocessed.toarray()

        feature_names = preprocessor.get_feature_names_out()

        # Compute SHAP
        _, shap_values = SHAPService.get_explainer_and_values(predictor, X_preprocessed)

        # Plot feature importance using custom horizontal bar chart (highly styled & stable)
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Create a DataFrame of importances
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": mean_abs_shap
        }).sort_values(by="importance", ascending=True).tail(15) # top 15

        fig, ax = plt.subplots(figsize=(8, 6))
        # Sleek color: dark blue
        bars = ax.barh(importance_df["feature"], importance_df["importance"], color="#3b82f6", edgecolor="#1e3a8a", height=0.6)
        
        # Grid and borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        plt.title("Global Feature Importance (Mean |SHAP Value|)", pad=15, fontsize=12, fontweight="bold")
        plt.xlabel("Mean Absolute SHAP Value (Impact on Model output)")
        plt.tight_layout()

        plot_dir = settings.plots_dir / str(dataset_id)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / "shap_summary.png"
        
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)

        return str(plot_path)

    @staticmethod
    def generate_local_explanation(
        dataset_id: int,
        prediction_id: int,
        model_payload: Dict[str, Any],
        raw_input_row: Dict[str, Any],
        predicted_class_idx: int = 0
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Computes local SHAP values for a single prediction row,
        saves a custom waterfall/impact bar plot, and returns raw contributions.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pipeline = model_payload["pipeline"]
        feature_cols = model_payload["feature_cols"]

        preprocessor = pipeline.named_steps["preprocessor"]
        predictor = pipeline.named_steps["predictor"]

        # Convert raw row to DataFrame
        row_df = pd.DataFrame([raw_input_row])
        # Subset to match training columns
        row_df = row_df[feature_cols]

        X_preprocessed = preprocessor.transform(row_df)
        if hasattr(X_preprocessed, "toarray"):
            X_preprocessed = X_preprocessed.toarray()

        feature_names = preprocessor.get_feature_names_out()

        # We need background data to initialize explainer
        # For local prediction, we can use a dummy background of the same row or a zero vector
        # since we just need the SHAP values for this single row.
        dummy_background = np.zeros((1, len(feature_names)))
        _, shap_values = SHAPService.get_explainer_and_values(predictor, X_preprocessed)

        # shap_values is now (1, n_features) or (n_features,)
        row_shap = shap_values[0] if len(shap_values.shape) > 1 else shap_values

        # Build list of contributions
        contributions = []
        for name, val in zip(feature_names, row_shap):
            contributions.append({
                "feature": str(name),
                "contribution": float(val)
            })

        # Generate a styled local bar plot
        # Sort by absolute contribution
        local_df = pd.DataFrame({
            "feature": feature_names,
            "contribution": row_shap
        })
        local_df["abs_contribution"] = local_df["contribution"].abs()
        local_df = local_df.sort_values(by="abs_contribution", ascending=True).tail(10) # top 10

        fig, ax = plt.subplots(figsize=(7, 5))
        
        # Colors: red for positive impact, blue for negative impact
        colors = ["#ef4444" if val >= 0 else "#3b82f6" for val in local_df["contribution"]]
        edge_colors = ["#991b1b" if val >= 0 else "#1e3a8a" for val in local_df["contribution"]]
        
        ax.barh(local_df["feature"], local_df["contribution"], color=colors, edgecolor=edge_colors, height=0.5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.axvline(0, color="#666666", linewidth=0.8, linestyle="-")

        plt.title("Feature Impact on Prediction (SHAP Values)", pad=15, fontsize=11, fontweight="bold")
        plt.xlabel("SHAP Value (Positive pushes prediction higher, Negative pushes lower)")
        plt.tight_layout()

        plot_dir = settings.plots_dir / str(dataset_id)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"prediction_{prediction_id}_waterfall.png"
        
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)

        # Sort contributions by absolute value descending for API response
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        return contributions, str(plot_path)
