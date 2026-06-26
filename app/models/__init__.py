"""
app/models/__init__.py
========================

Importing this package imports every ORM model module, which registers each
model's table on `Base.metadata`. This must happen before `init_db()` calls
`Base.metadata.create_all(...)`, otherwise tables for unimported models
simply won't be created.

Import models from here elsewhere in the codebase, e.g.:

    from app.models import Dataset, TrainingJob, TrainedModel
"""

from app.models.dataset import Dataset
from app.models.training_job import TrainingJob
from app.models.trained_model import TrainedModel
from app.models.prediction import Prediction
from app.models.explanation import Explanation
from app.models.report import Report

__all__ = [
    "Dataset",
    "TrainingJob",
    "TrainedModel",
    "Prediction",
    "Explanation",
    "Report",
]
