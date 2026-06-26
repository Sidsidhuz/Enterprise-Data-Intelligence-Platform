from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session
from fastapi import HTTPException

from app.config import settings
from app.models.dataset import Dataset


class ValidationService:
    @staticmethod
    def validate_dataset(db: Session, dataset: Dataset) -> pd.DataFrame:
        """
        Loads the dataset from disk using Pandas.
        Checks for empty files, structural issues, and malformed rows.
        Updates row/column count and updates status to 'validated'.
        Returns the loaded pandas DataFrame.
        """
        absolute_path = settings.data_path / dataset.storage_path
        if not absolute_path.exists():
            dataset.status = "failed"
            db.commit()
            raise HTTPException(
                status_code=404,
                detail=f"Dataset file not found at {dataset.storage_path}",
            )

        # Detect extension and read
        try:
            if dataset.storage_path.endswith(".csv"):
                # Try reading a few lines first to catch delimiter errors or empty files
                df = pd.read_csv(absolute_path)
            elif dataset.storage_path.endswith((".xlsx", ".xls")):
                df = pd.read_excel(absolute_path)
            else:
                raise ValueError("Unsupported file format in storage path")
        except Exception as e:
            dataset.status = "failed"
            db.commit()
            raise HTTPException(
                status_code=400,
                detail=f"File could not be parsed: {str(e)}",
            )

        # Structural checks
        rows, cols = df.shape
        if rows == 0:
            dataset.status = "failed"
            db.commit()
            raise HTTPException(
                status_code=400,
                detail="Dataset is empty (has 0 rows).",
            )
        if cols == 0:
            dataset.status = "failed"
            db.commit()
            raise HTTPException(
                status_code=400,
                detail="Dataset has 0 columns.",
            )

        # Header check: check if columns have empty or unnamed headers
        unnamed_cols = [col for col in df.columns if "Unnamed:" in str(col)]
        if len(unnamed_cols) == len(df.columns):
            dataset.status = "failed"
            db.commit()
            raise HTTPException(
                status_code=400,
                detail="Missing header row: all column names are auto-generated/unnamed.",
            )

        # Update dataset record
        dataset.rows = rows
        dataset.columns = cols
        dataset.status = "validated"
        db.commit()
        db.refresh(dataset)

        return df
