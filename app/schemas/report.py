from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class ReportRequest(BaseModel):
    report_type: str  # "pdf" or "excel"


class ReportResponse(BaseModel):
    id: int
    dataset_id: int
    report_type: str
    status: str  # generating, completed, failed
    storage_path: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True
