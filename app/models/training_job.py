"""
app/models/training_job.py
============================

Represents a single AutoML training run against a dataset: which column was
targeted, what kind of problem it is, which algorithms were requested, and
how the run progressed.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)

    target_column: Mapped[str] = mapped_column(String(255), nullable=False)

    # "classification" or "regression" — inferred automatically, but stored
    # explicitly since the user may override the inference.
    problem_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # queued -> running -> completed -> failed
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="queued")

    # JSON-as-text: requested algorithms, tuning budget, etc.
    config_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="training_jobs")
    models: Mapped[list["TrainedModel"]] = relationship(
        "TrainedModel", back_populates="training_job", cascade="all, delete-orphan"
    )

    def get_config(self) -> dict[str, Any]:
        if not self.config_json:
            return {}
        return json.loads(self.config_json)

    def set_config(self, config: dict[str, Any]) -> None:
        self.config_json = json.dumps(config)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<TrainingJob id={self.id} dataset_id={self.dataset_id} "
            f"target={self.target_column!r} status={self.status!r}>"
        )
