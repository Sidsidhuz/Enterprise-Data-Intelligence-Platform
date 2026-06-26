from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.trained_model import TrainedModel
from app.schemas.trained_model import TrainedModelResponse
from app.schemas.prediction import PredictionRequest, PredictionResponse, ExplanationResponse, SHAPContribution
from app.modules.prediction.predict_service import PredictionService

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/{model_id}", response_model=TrainedModelResponse)
def get_model(model_id: int, db: Session = Depends(get_db)):
    """Retrieves metadata, hyperparameters, and evaluation metrics for a specific model."""
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
        
    return TrainedModelResponse(
        id=model.id,
        training_job_id=model.training_job_id,
        algorithm=model.algorithm,
        hyperparameters=model.get_hyperparameters(),
        metrics=model.get_metrics(),
        primary_metric_value=model.primary_metric_value,
        is_best=model.is_best,
        created_at=model.created_at,
    )


@router.post("/{model_id}/predict", response_model=PredictionResponse)
def predict(
    model_id: int, request: PredictionRequest, db: Session = Depends(get_db)
):
    """
    Serves a prediction for a single data record.
    Includes SHAP feature attribution explainability.
    """
    prediction_record, explanation_record = PredictionService.predict_single(
        db, model_id, request.input_data
    )

    # Build Pydantic response
    output = prediction_record.get_output()
    
    explanation_res = None
    if explanation_record:
        contribs = [
            SHAPContribution(feature=x["feature"], contribution=x["contribution"])
            for x in explanation_record.get_shap_values()
        ]
        explanation_res = ExplanationResponse(
            shap_values=contribs,
            plot_path=explanation_record.plot_path,
        )

    return PredictionResponse(
        prediction_id=prediction_record.id,
        model_id=model_id,
        prediction=output["prediction"],
        probability=output.get("probability"),
        explanation=explanation_res,
        created_at=prediction_record.created_at,
    )


@router.post("/{model_id}/predict-batch")
def predict_batch(
    model_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)
):
    """
    Accepts an uploaded CSV batch file, generates predictions for each row,
    and returns a downloadable CSV with predictions appended.
    """
    # Verify model exists
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files supported for batch prediction.")

    # Save uploaded batch file temporarily
    temp_in = NamedTemporaryFile(delete=False, suffix=".csv")
    temp_out_path = Path(temp_in.name).parent / f"batch_preds_{model_id}.csv"
    
    try:
        # Write upload to temp file
        contents = file.file.read()
        temp_in.write(contents)
        temp_in.close()

        # Generate predictions
        df_out = PredictionService.predict_batch(db, model_id, Path(temp_in.name))
        
        # Save output to file
        df_out.to_csv(temp_out_path, index=False)
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Batch prediction failure: {str(e)}")
    finally:
        # Clean up input temp file
        if os.path.exists(temp_in.name):
            os.remove(temp_in.name)

    # Return prediction file response and set cleanup on close if needed
    # Note: FileResponse will stream the file. We can delete it afterwards, or let the OS clean up temp folder.
    return FileResponse(
        path=str(temp_out_path),
        filename=f"batch_predictions_{model.algorithm}.csv",
        media_type="text/csv",
    )
