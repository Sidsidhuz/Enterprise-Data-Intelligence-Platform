from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from app.models.dataset import Dataset
from app.config import settings


class EDAService:
    @staticmethod
    def get_correlation_matrix(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates correlation matrix for all numeric columns.
        Returns a dict with columns list and a 2D matrix of values.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {"columns": [], "matrix": []}
            
        corr = numeric_df.corr().fillna(0.0)
        return {
            "columns": corr.columns.tolist(),
            "matrix": corr.values.tolist(),
        }

    @staticmethod
    def get_distributions(df: pd.DataFrame, dtypes: Dict[str, str]) -> Dict[str, Any]:
        """
        Calculates distribution data for columns:
        - Numeric: Bins and counts.
        - Categorical/Boolean: Value counts (top 15).
        """
        distributions = {}

        for col in df.columns:
            col_type = dtypes.get(col, "categorical")
            series = df[col].dropna()
            if series.empty:
                continue

            if col_type == "numeric":
                # Compute histograms using numpy
                counts, bin_edges = np.histogram(series, bins=10)
                distributions[col] = {
                    "type": "numeric",
                    "counts": counts.tolist(),
                    "bin_edges": bin_edges.tolist(),
                }
            else:
                # Value counts for categorical/bool
                val_counts = series.value_counts().head(15)
                distributions[col] = {
                    "type": "categorical",
                    "categories": val_counts.index.map(str).tolist(),
                    "counts": val_counts.values.tolist(),
                }

        return distributions

    @staticmethod
    def generate_static_plots(dataset_id: int, df: pd.DataFrame) -> List[str]:
        """
        Generates and saves static visualization plots to data/plots/{dataset_id}/
        Returns a list of saved absolute paths.
        """
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        plot_paths = []
        plot_dir = settings.plots_dir / str(dataset_id)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # 1. Correlation Heatmap
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty and numeric_df.shape[1] > 1:
            corr = numeric_df.corr().fillna(0.0)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            fig.colorbar(cax)
            
            # Set labels
            ax.set_xticks(np.arange(len(corr.columns)))
            ax.set_yticks(np.arange(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="left")
            ax.set_yticklabels(corr.columns)
            
            plt.title("Correlation Heatmap", pad=20)
            plt.tight_layout()
            
            heatmap_path = plot_dir / "correlation_heatmap.png"
            plt.savefig(heatmap_path, dpi=150)
            plt.close(fig)
            plot_paths.append(str(heatmap_path))

        return plot_paths
