"""
app/models/explanation.py
============================

Stores the SHAP explanation associated with a single prediction. Kept as a
separate table (rather than columns on Prediction) because explanations are
optional, larger, and conceptually a distinct concern — this also makes it
trivial to add other explanation types later (e.g. LIME) without altering
the predictions table.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Explanation(Base):
    __tablename__ = "explanations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), nullable=False, unique=True)

    # JSON array of {"feature": ..., "shap_value": ...} entries.
    shap_values_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Path to a saved waterfall plot image, relative to /data, e.g.
    # "plots/1/prediction_42_waterfall.png". Optional — some callers only
    # need the raw SHAP values (e.g. the API), not a rendered image.
    plot_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    prediction: Mapped["Prediction"] = relationship("Prediction", back_populates="explanation")

    def get_shap_values(self) -> list[dict[str, Any]]:
        return json.loads(self.shap_values_json)

    def set_shap_values(self, values: list[dict[str, Any]]) -> None:
        self.shap_values_json = json.dumps(values)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Explanation id={self.id} prediction_id={self.prediction_id}>"
