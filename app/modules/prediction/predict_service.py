from __future__ import annotations

import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from fastapi import HTTPException
from typing import Dict, Any, List, Tuple

from app.config import settings
from app.models.trained_model import TrainedModel
from app.models.prediction import Prediction
from app.models.explanation import Explanation
from app.modules.explainability.shap_service import SHAPService
from app.modules.feature_engineering.feature_service import FeatureService


class PredictionService:
    @staticmethod
    def load_model_payload(model_id: int) -> Dict[str, Any]:
        """Loads model payload (pipeline, feature columns, metadata) from disk."""
        model_dir = settings.models_dir / str(model_id)
        artifact_path = model_dir / "model.joblib"
        if not artifact_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model artifact not found for model ID {model_id}",
            )
        try:
            payload = joblib.load(artifact_path)
            return payload
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model artifact: {str(e)}",
            )

    @staticmethod
    def predict_single(
        db: Session, model_id: int, input_data: Dict[str, Any]
    ) -> Tuple[Prediction, Explanation]:
        """
        Runs single instance inference using the specified model.
        Computes SHAP values, creates DB records for prediction and explanation.
        """
        model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        # Load payload
        payload = PredictionService.load_model_payload(model_id)
        pipeline = payload["pipeline"]
        feature_cols = payload["feature_cols"]
        problem_type = payload["problem_type"]
        target_col = payload["target_col"]

        # Parse input row into Pandas
        row_dict = {}
        for col in feature_cols:
            # Handle possible datetime decompositions (e.g. col_year, col_month)
            if col in input_data:
                row_dict[col] = input_data[col]
            else:
                # If feature is a decomposed datetime and the input contains the base column
                base_col = col.rsplit("_", 1)[0]
                if base_col in input_data and col.rsplit("_", 1)[1] in ["year", "month", "day", "dayofweek"]:
                    try:
                        dt = pd.to_datetime(input_data[base_col])
                        suffix = col.rsplit("_", 1)[1]
                        if suffix == "year":
                            row_dict[col] = dt.year
                        elif suffix == "month":
                            row_dict[col] = dt.month
                        elif suffix == "day":
                            row_dict[col] = dt.day
                        elif suffix == "dayofweek":
                            row_dict[col] = dt.dayofweek
                    except Exception:
                        row_dict[col] = np.nan
                else:
                    row_dict[col] = np.nan

        input_df = pd.DataFrame([row_dict])

        # Run Prediction
        try:
            pred_arr = pipeline.predict(input_df)
            prediction_val = pred_arr[0]

            probability_val = None
            predicted_class_idx = 0

            if problem_type == "classification":
                # Ensure values are serializable
                if isinstance(prediction_val, (np.integer, np.int64)):
                    prediction_val = int(prediction_val)
                elif isinstance(prediction_val, (np.floating, np.float64)):
                    prediction_val = float(prediction_val)
                elif isinstance(prediction_val, np.bool_):
                    prediction_val = bool(prediction_val)
                
                # Get prediction probabilities
                if hasattr(pipeline, "predict_proba"):
                    probs = pipeline.predict_proba(input_df)[0]
                    # Probability of the predicted class
                    predicted_class_idx = int(np.argmax(probs))
                    probability_val = float(probs[predicted_class_idx])
            else:
                prediction_val = float(prediction_val)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Inference failure: {str(e)}",
            )

        # Build output structure
        output_payload = {
            "prediction": prediction_val,
            "probability": probability_val,
        }

        # Create Prediction record in DB (must commit to get prediction.id for SHAP plots)
        prediction_record = Prediction(
            model_id=model_id,
        )
        prediction_record.set_input(input_data)
        prediction_record.set_output(output_payload)
        
        db.add(prediction_record)
        db.commit()
        db.refresh(prediction_record)

        # Get dataset_id from the job
        dataset_id = model.training_job.dataset_id

        # Compute SHAP explanation
        try:
            contributions, plot_path = SHAPService.generate_local_explanation(
                dataset_id,
                prediction_record.id,
                payload,
                row_dict,
                predicted_class_idx
            )
            
            # Save Explanation record
            # Storage plot path is relative
            relative_plot_path = f"plots/{dataset_id}/prediction_{prediction_record.id}_waterfall.png"
            
            explanation_record = Explanation(
                prediction_id=prediction_record.id,
                plot_path=relative_plot_path,
            )
            explanation_record.set_shap_values(contributions)
            
            db.add(explanation_record)
            db.commit()
            db.refresh(explanation_record)

        except Exception as e:
            # Create a fallback empty explanation on failure so prediction is still served
            explanation_record = Explanation(
                prediction_id=prediction_record.id,
                plot_path=None,
            )
            explanation_record.set_shap_values([])
            db.add(explanation_record)
            db.commit()
            db.refresh(explanation_record)
            print(f"SHAP generation failed for prediction {prediction_record.id}: {str(e)}")

        return prediction_record, explanation_record

    @staticmethod
    def predict_batch(
        db: Session, model_id: int, file_path: Path
    ) -> pd.DataFrame:
        """
        Runs batch predictions on a CSV upload file.
        Returns a pandas DataFrame with predictions (and probabilities if classification) appended.
        """
        payload = PredictionService.load_model_payload(model_id)
        pipeline = payload["pipeline"]
        feature_cols = payload["feature_cols"]
        problem_type = payload["problem_type"]
        target_col = payload["target_col"]
        dtypes = payload["dtypes"]

        # Read batch data
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not parse batch CSV: {str(e)}"
            )

        # Decompose datetimes in dataframe if needed
        df_prepped, _ = FeatureService.preprocess_dataframe_pandas(df, target_col, dtypes)

        # Align columns (fill missing with NaN)
        for col in feature_cols:
            if col not in df_prepped.columns:
                df_prepped[col] = np.nan

        # Subset features
        X_batch = df_prepped[feature_cols]

        try:
            preds = pipeline.predict(X_batch)
            df["prediction"] = preds

            if problem_type == "classification" and hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(X_batch)
                # Save max probability
                df["probability"] = np.max(probs, axis=1)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Batch inference failed: {str(e)}"
            )

        return df
