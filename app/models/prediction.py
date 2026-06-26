"""
app/models/prediction.py
==========================

An append-only log of every prediction served by the application. Storing
these (rather than discarding them after the response is sent) lets the UI
show prediction history and lets the SHAP explanation be looked up later
without recomputing it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False)

    input_json: Mapped[str] = mapped_column(Text, nullable=False)
    output_json: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    model: Mapped["TrainedModel"] = relationship("TrainedModel", back_populates="predictions")
    explanation: Mapped[Optional["Explanation"]] = relationship(
        "Explanation", back_populates="prediction", uselist=False, cascade="all, delete-orphan"
    )

    def get_input(self) -> dict[str, Any]:
        return json.loads(self.input_json)

    def set_input(self, value: dict[str, Any]) -> None:
        self.input_json = json.dumps(value)

    def get_output(self) -> dict[str, Any]:
        return json.loads(self.output_json)

    def set_output(self, value: dict[str, Any]) -> None:
        self.output_json = json.dumps(value)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Prediction id={self.id} model_id={self.model_id}>"
