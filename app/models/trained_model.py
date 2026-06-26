"""
app/models/trained_model.py
=============================

Represents a single trained model candidate produced by a TrainingJob.
A training job typically produces several of these (one per algorithm);
exactly one should have `is_best = True`.

`artifact_path` points to the serialized (joblib) model + preprocessing
pipeline on disk, relative to the project's /data directory, e.g.
"models/3/model.joblib".
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class TrainedModel(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    training_job_id: Mapped[int] = mapped_column(ForeignKey("training_jobs.id"), nullable=False)

    # e.g. "xgboost", "random_forest", "logistic_regression"
    algorithm: Mapped[str] = mapped_column(String(100), nullable=False)

    hyperparameters_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Primary sort metric, duplicated out of metrics_json for fast SQL ORDER BY
    # (e.g. ROC-AUC for classification, R^2 for regression).
    primary_metric_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    artifact_path: Mapped[str] = mapped_column(String(500), nullable=False)
    is_best: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    training_job: Mapped["TrainingJob"] = relationship("TrainingJob", back_populates="models")
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction", back_populates="model", cascade="all, delete-orphan"
    )

    def get_hyperparameters(self) -> dict[str, Any]:
        return json.loads(self.hyperparameters_json) if self.hyperparameters_json else {}

    def set_hyperparameters(self, params: dict[str, Any]) -> None:
        self.hyperparameters_json = json.dumps(params)

    def get_metrics(self) -> dict[str, Any]:
        return json.loads(self.metrics_json) if self.metrics_json else {}

    def set_metrics(self, metrics: dict[str, Any]) -> None:
        self.metrics_json = json.dumps(metrics)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<TrainedModel id={self.id} algorithm={self.algorithm!r} is_best={self.is_best}>"
