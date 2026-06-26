from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session
from fastapi import HTTPException
from typing import Dict, Any, Optional

from app.models.dataset import Dataset
from app.config import settings


class CleaningService:
    @staticmethod
    def clean_dataset(
        db: Session,
        dataset: Dataset,
        df: pd.DataFrame,
        imputation_overrides: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Deduplicates rows, imputes missing values based on defaults or user overrides,
        drops 100% null columns, and saves the cleaned dataset.
        Updates status to 'cleaned' in the DB.
        """
        if imputation_overrides is None:
            imputation_overrides = {}

        # 1. Drop duplicate rows
        df_cleaned = df.drop_duplicates()

        # Get schema metadata to know data types
        metadata = dataset.get_schema_metadata()
        if not metadata:
            raise HTTPException(
                status_code=400,
                detail="Dataset has not been profiled yet. Please profile before cleaning.",
            )
        
        dtypes = metadata.get("dtypes", {})

        # 2. Drop 100% null columns
        cols_to_drop = []
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().all():
                cols_to_drop.append(col)
        
        if cols_to_drop:
            df_cleaned = df_cleaned.drop(columns=cols_to_drop)

        # 3. Imputation
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() == 0:
                continue

            col_type = dtypes.get(col, "categorical")
            strategy = imputation_overrides.get(col)

            if col_type == "numeric":
                # Default is median
                if strategy == "mean":
                    val = df_cleaned[col].mean()
                elif strategy == "constant":
                    val = 0.0  # fallback constant for numeric
                else:  # median
                    val = df_cleaned[col].median()
                
                # If everything in the column was null (which shouldn't happen because we dropped all-null cols), val could be nan
                if pd.isna(val):
                    val = 0.0
                df_cleaned[col] = df_cleaned[col].fillna(val)

            else:  # categorical or boolean
                # Default is mode
                if strategy == "constant":
                    val = "Unknown"
                else:  # mode
                    mode_series = df_cleaned[col].mode()
                    val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                
                df_cleaned[col] = df_cleaned[col].fillna(val)

        # Ensure cleaned directories exist
        cleaned_dir = settings.cleaned_dir / str(dataset.id)
        cleaned_dir.mkdir(parents=True, exist_ok=True)

        relative_path = f"cleaned/{dataset.id}/cleaned.csv"
        absolute_path = settings.data_path / relative_path

        # Save cleaned dataset as CSV
        df_cleaned.to_csv(absolute_path, index=False)

        # Update dataset DB status and storage details
        dataset.status = "cleaned"
        db.commit()
        db.refresh(dataset)

        return df_cleaned
