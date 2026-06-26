from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class TrainingJobCreate(BaseModel):
    target_column: str
    problem_type: Optional[str] = None  # classification or regression, can be auto-inferred
    algorithms: Optional[List[str]] = None  # list of algorithms to try, e.g. ["xgboost", "random_forest"]
    tuning_budget_seconds: Optional[int] = 60


class LeaderboardItem(BaseModel):
    model_id: int
    algorithm: str
    metrics: Dict[str, float]
    primary_metric_value: Optional[float]
    is_best: bool
    created_at: datetime


class TrainingJobResponse(BaseModel):
    id: int
    dataset_id: int
    target_column: str
    problem_type: str
    status: str  # queued, running, completed, failed
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    leaderboard: List[LeaderboardItem] = []

    class Config:
        from_attributes = True
