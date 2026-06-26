from __future__ import annotations

import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.dataset import Dataset
from app.schemas.dataset import DatasetResponse, DatasetProfileResponse
from app.modules.upload.upload_service import UploadService
from app.modules.validation.validation_service import ValidationService
from app.modules.profiling.profiling_service import ProfilingService
from app.modules.cleaning.cleaning_service import CleaningService
from app.modules.eda.eda_service import EDAService

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("", response_model=DatasetResponse, status_code=201)
def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Uploads a dataset file, validates it structurally, and auto-generates
    the data profile, returning the dataset metadata.
    """
    # 1. Handle Upload
    dataset = UploadService.handle_upload(db, file)
    
    # 2. Run Validation & Profile synchronously to prepare data preview
    try:
        df = ValidationService.validate_dataset(db, dataset)
        ProfilingService.profile_dataset(db, dataset, df)
    except Exception as e:
        # Status is marked failed inside the services
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=f"Validation/Profiling failed: {str(e)}")

    return dataset


@router.get("", response_model=List[DatasetResponse])
def list_datasets(db: Session = Depends(get_db)):
    """Lists history of all uploaded datasets."""
    return db.query(Dataset).order_by(Dataset.uploaded_at.desc()).all()


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Retrieves a single dataset's metadata."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.get("/{dataset_id}/profile", response_model=DatasetProfileResponse)
def get_dataset_profile(dataset_id: int, db: Session = Depends(get_db)):
    """Returns the pre-computed statistical profile of the dataset."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    metadata = dataset.get_schema_metadata()
    if not metadata or dataset.status in ["uploaded", "validated"]:
        raise HTTPException(
            status_code=400,
            detail="Profile not generated yet. Try profiling first.",
        )
    return metadata


@router.post("/{dataset_id}/clean", response_model=DatasetResponse)
def clean_dataset(
    dataset_id: int,
    imputation_overrides: Optional[Dict[str, str]] = None,
    db: Session = Depends(get_db),
):
    """
    Performs data cleaning operations (imputation, deduplication) on the dataset.
    Accepts user overrides for specific column imputation strategies.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        # Load raw data to clean
        df = ValidationService.validate_dataset(db, dataset)
        CleaningService.clean_dataset(db, dataset, df, imputation_overrides)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=f"Cleaning failed: {str(e)}")

    return dataset


@router.get("/{dataset_id}/eda")
def get_dataset_eda(dataset_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Computes and returns JSON-formatted data for interactive Plotly charts:
    correlation heatmap values and column distributions.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset.status != "cleaned":
        raise HTTPException(
            status_code=400,
            detail="Dataset has not been cleaned yet. Clean dataset to compute EDA data.",
        )

    # Load cleaned data
    import pandas as pd
    from app.config import settings
    cleaned_path = settings.data_path / f"cleaned/{dataset.id}/cleaned.csv"
    if not cleaned_path.exists():
        raise HTTPException(status_code=404, detail="Cleaned dataset file not found.")

    df = pd.read_csv(cleaned_path)
    metadata = dataset.get_schema_metadata()
    dtypes = metadata.get("dtypes", {})

    # Compute correlation and distributions
    corr_data = EDAService.get_correlation_matrix(df)
    dist_data = EDAService.get_distributions(df, dtypes)

    return {
        "dataset_id": dataset_id,
        "correlation": corr_data,
        "distributions": dist_data,
    }
