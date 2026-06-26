from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel


class TrainedModelResponse(BaseModel):
    id: int
    training_job_id: int
    algorithm: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    primary_metric_value: Optional[float]
    is_best: bool
    created_at: datetime

    class Config:
        from_attributes = True
