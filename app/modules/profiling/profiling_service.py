from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from typing import Dict, Any

from app.models.dataset import Dataset
from app.config import settings


class ProfilingService:
    @staticmethod
    def profile_dataset(db: Session, dataset: Dataset, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates profiling metrics: missing percentages, data types, duplicates, basic stats,
        and outlier counts. Updates schema_metadata_json and status in DB.
        """
        row_count, col_count = df.shape
        
        # Calculate duplicate rows
        duplicate_count = int(df.duplicated().sum())

        # Dtypes inference and missing values
        missing_values: Dict[str, str] = {}
        dtypes: Dict[str, str] = {}
        outlier_counts: Dict[str, int] = {}
        summary_stats: Dict[str, Dict[str, Any]] = {}

        for col in df.columns:
            # Missing value percentage
            missing_count = int(df[col].isnull().sum())
            missing_pct = (missing_count / row_count) * 100
            missing_values[col] = f"{missing_pct:.1f}%"

            # Infer data types
            col_series = df[col]
            if pd.api.types.is_numeric_dtype(col_series):
                if col_series.nunique() <= 2:
                    dtypes[col] = "boolean"
                elif pd.api.types.is_integer_dtype(col_series) and col_series.nunique() <= 20:
                    dtypes[col] = "categorical"  # low cardinality integer
                else:
                    dtypes[col] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(col_series):
                dtypes[col] = "datetime"
            elif isinstance(col_series.dtype, pd.CategoricalDtype):
                dtypes[col] = "categorical"
            elif col_series.dtype == "bool":
                dtypes[col] = "boolean"
            else:
                # Try parsing as datetime if string
                try:
                    pd.to_datetime(col_series.dropna().head(100), errors="raise")
                    dtypes[col] = "datetime"
                except (ValueError, TypeError):
                    # Check unique values
                    if col_series.nunique() <= 2:
                        dtypes[col] = "boolean"
                    else:
                        dtypes[col] = "categorical"

            # Compute stats and outliers
            outlier_counts[col] = 0
            col_stats: Dict[str, Any] = {}

            # Drop missing for calculations
            clean_series = col_series.dropna()
            
            if dtypes[col] == "numeric":
                # Compute numeric stats
                q1 = float(clean_series.quantile(0.25)) if not clean_series.empty else 0.0
                q3 = float(clean_series.quantile(0.75)) if not clean_series.empty else 0.0
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Outliers count
                if not clean_series.empty:
                    outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
                    outlier_counts[col] = int(outliers.count())
                
                col_stats = {
                    "min": float(clean_series.min()) if not clean_series.empty else 0.0,
                    "max": float(clean_series.max()) if not clean_series.empty else 0.0,
                    "mean": float(clean_series.mean()) if not clean_series.empty else 0.0,
                    "median": float(clean_series.median()) if not clean_series.empty else 0.0,
                    "std": float(clean_series.std()) if not clean_series.empty else 0.0,
                    "q1": q1,
                    "q3": q3,
                    "unique_count": int(col_series.nunique()),
                }
            else:
                # Compute categorical/boolean stats
                col_stats = {
                    "unique_count": int(col_series.nunique()),
                    "mode": str(col_series.mode().iloc[0]) if not col_series.mode().empty else "N/A",
                }

            summary_stats[col] = col_stats

        # Bundle everything into a single metadata schema
        profile_data = {
            "dataset_id": dataset.id,
            "missing_values": missing_values,
            "duplicates": duplicate_count,
            "dtypes": dtypes,
            "row_count": row_count,
            "column_count": col_count,
            "summary_stats": summary_stats,
            "outlier_counts": outlier_counts,
        }

        # Save to DB
        dataset.set_schema_metadata(profile_data)
        dataset.status = "profiled"
        db.commit()
        db.refresh(dataset)

        return profile_data
