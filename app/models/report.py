"""
app/models/report.py
======================

Represents a generated report (PDF or Excel) for a dataset, bundling
profiling, EDA, leaderboard, and SHAP results into a single downloadable
file stored under /data/reports.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)

    # "pdf" or "excel"
    report_type: Mapped[str] = mapped_column(String(20), nullable=False)

    # generating -> completed -> failed
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="generating")

    storage_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="reports")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Report id={self.id} dataset_id={self.dataset_id} type={self.report_type!r}>"
