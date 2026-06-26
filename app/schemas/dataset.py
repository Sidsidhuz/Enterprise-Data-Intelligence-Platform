from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel


class DatasetBase(BaseModel):
    filename: str


class DatasetCreate(DatasetBase):
    pass


class DatasetResponse(DatasetBase):
    id: int
    storage_path: str
    status: str
    rows: Optional[int] = None
    columns: Optional[int] = None
    uploaded_at: datetime

    class Config:
        from_attributes = True


class DatasetProfileResponse(BaseModel):
    dataset_id: int
    missing_values: Dict[str, str]  # e.g., {"income": "4.2%"}
    duplicates: int
    dtypes: Dict[str, str]  # e.g., {"income": "float64"}
    row_count: int
    column_count: int
    summary_stats: Dict[str, Dict[str, Any]]  # column -> stat -> value
    outlier_counts: Dict[str, int]
