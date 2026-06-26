from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    input_data: Dict[str, Any]


class SHAPContribution(BaseModel):
    feature: str
    contribution: float  # SHAP value


class ExplanationResponse(BaseModel):
    shap_values: List[SHAPContribution]
    plot_path: Optional[str] = None


class PredictionResponse(BaseModel):
    prediction_id: int
    model_id: int
    prediction: Any  # Can be float (regression) or str/int/bool (classification)
    probability: Optional[float] = None  # Prob of positive class / highest class if classification
    explanation: Optional[ExplanationResponse] = None
    created_at: datetime
