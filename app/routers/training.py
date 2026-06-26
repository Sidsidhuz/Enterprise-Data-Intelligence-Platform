from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.dataset import Dataset
from app.models.training_job import TrainingJob
from app.models.trained_model import TrainedModel
from app.schemas.training_job import TrainingJobCreate, TrainingJobResponse, LeaderboardItem
from app.modules.automl.automl_service import AutoMLService

router = APIRouter(tags=["training"])


@router.get("/training-jobs", response_model=List[TrainingJobResponse])
def list_training_jobs(
    dataset_id: Optional[int] = Query(default=None, description="Filter by dataset ID"),
    db: Session = Depends(get_db),
):
    """Lists all training jobs, optionally filtered by dataset_id."""
    query = db.query(TrainingJob)
    if dataset_id is not None:
        query = query.filter(TrainingJob.dataset_id == dataset_id)
    jobs = query.order_by(TrainingJob.id.desc()).all()

    result = []
    for job in jobs:
        leaderboard: List[LeaderboardItem] = []
        sorted_models = sorted(job.models, key=lambda m: m.primary_metric_value or 0.0, reverse=True)
        for m in sorted_models:
            leaderboard.append(
                LeaderboardItem(
                    model_id=m.id,
                    algorithm=m.algorithm,
                    metrics=m.get_metrics(),
                    primary_metric_value=m.primary_metric_value,
                    is_best=m.is_best,
                    created_at=m.created_at,
                )
            )
        result.append(
            TrainingJobResponse(
                id=job.id,
                dataset_id=job.dataset_id,
                target_column=job.target_column,
                problem_type=job.problem_type,
                status=job.status,
                error_message=job.error_message,
                started_at=job.started_at,
                completed_at=job.completed_at,
                leaderboard=leaderboard,
            )
        )
    return result


@router.post("/datasets/{dataset_id}/train", response_model=TrainingJobResponse, status_code=202)
def start_training(
    dataset_id: int,
    config: TrainingJobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Submits a training job for the specified dataset.
    The job runs asynchronously in the background.
    """
    # Verify dataset exists and is cleaned
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if dataset.status != "cleaned":
        raise HTTPException(
            status_code=400,
            detail="Dataset must be cleaned before training. Please clean the dataset first.",
        )

    # Validate target column exists
    metadata = dataset.get_schema_metadata()
    dtypes = metadata.get("dtypes", {})
    if config.target_column not in dtypes:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{config.target_column}' does not exist in the dataset schema.",
        )

    # Check if a training job is already running for this dataset
    active_job = (
        db.query(TrainingJob)
        .filter(TrainingJob.dataset_id == dataset_id, TrainingJob.status.in_(["queued", "running"]))
        .first()
    )
    if active_job:
        raise HTTPException(
            status_code=409,
            detail=f"A training job is already running for this dataset (Job ID: {active_job.id}).",
        )

    # Create job config dict
    job_config = {
        "algorithms": config.algorithms,
        "tuning_budget_seconds": config.tuning_budget_seconds,
    }

    # Inferred problem type if not specified
    problem_type = config.problem_type
    if not problem_type:
        # We need to temporarily load data to check cardinality or infer
        import pandas as pd
        from app.config import settings
        cleaned_path = settings.data_path / f"cleaned/{dataset_id}/cleaned.csv"
        try:
            df = pd.read_csv(cleaned_path)
            problem_type = AutoMLService.infer_problem_type(df, config.target_column)
        except Exception:
            problem_type = "classification"  # safe fallback

    # Create TrainingJob record
    job = TrainingJob(
        dataset_id=dataset_id,
        target_column=config.target_column,
        problem_type=problem_type,
        status="queued",
    )
    job.set_config(job_config)
    
    db.add(job)
    db.commit()
    db.refresh(job)

    # Submit task to BackgroundTasks
    background_tasks.add_task(AutoMLService.run_training_job_sync, db, job.id)

    # Map to response (empty leaderboard initially)
    return TrainingJobResponse(
        id=job.id,
        dataset_id=job.dataset_id,
        target_column=job.target_column,
        problem_type=job.problem_type,
        status=job.status,
        started_at=job.started_at,
        completed_at=job.completed_at,
        leaderboard=[],
    )


@router.get("/training-jobs/{job_id}", response_model=TrainingJobResponse)
def get_training_job(job_id: int, db: Session = Depends(get_db)):
    """Retrieves the status and performance leaderboard of a training job."""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    # Construct Leaderboard list
    leaderboard: List[LeaderboardItem] = []
    
    # Sort models by metric value descending
    sorted_models = sorted(job.models, key=lambda m: m.primary_metric_value or 0.0, reverse=True)
    
    for m in sorted_models:
        leaderboard.append(
            LeaderboardItem(
                model_id=m.id,
                algorithm=m.algorithm,
                metrics=m.get_metrics(),
                primary_metric_value=m.primary_metric_value,
                is_best=m.is_best,
                created_at=m.created_at,
            )
        )

    # Build response
    res = TrainingJobResponse(
        id=job.id,
        dataset_id=job.dataset_id,
        target_column=job.target_column,
        problem_type=job.problem_type,
        status=job.status,
        error_message=job.error_message,
        started_at=job.started_at,
        completed_at=job.completed_at,
        leaderboard=leaderboard,
    )
    return res
