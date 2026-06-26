"""
app/models/dataset.py
======================

Represents a single uploaded dataset and its lifecycle status.

`schema_metadata_json` stores a JSON-serialized dict describing each
column's inferred dtype and basic stats, e.g.:

    {"income": {"dtype": "float64", "nullable": true},
     "region": {"dtype": "object", "nullable": false}}

SQLite has no native JSON column type, so this is stored as TEXT and
serialized/deserialized in Python using the helpers below.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Original filename as uploaded by the user (display purposes only —
    # never used directly as a filesystem path; see app/modules/upload).
    filename: Mapped[str] = mapped_column(String(255), nullable=False)

    # Path to the stored raw file, relative to the project's /data directory,
    # e.g. "raw/1/original.csv".
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)

    # Lifecycle status: uploaded -> validated -> profiled -> cleaned -> failed
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="uploaded")

    rows: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    columns: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # JSON-as-text column metadata (see module docstring).
    schema_metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # One dataset can have many training jobs and many reports.
    training_jobs: Mapped[list["TrainingJob"]] = relationship(
        "TrainingJob", back_populates="dataset", cascade="all, delete-orphan"
    )
    reports: Mapped[list["Report"]] = relationship(
        "Report", back_populates="dataset", cascade="all, delete-orphan"
    )

    # ------------------------------------------------------------------
    # Convenience helpers for the JSON-as-text column
    # ------------------------------------------------------------------

    def get_schema_metadata(self) -> dict[str, Any]:
        """Deserialize `schema_metadata_json` into a dict (empty if unset)."""
        if not self.schema_metadata_json:
            return {}
        return json.loads(self.schema_metadata_json)

    def set_schema_metadata(self, metadata: dict[str, Any]) -> None:
        """Serialize a dict into `schema_metadata_json`."""
        self.schema_metadata_json = json.dumps(metadata)

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return f"<Dataset id={self.id} filename={self.filename!r} status={self.status!r}>"
