from __future__ import annotations

import os
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.dataset import Dataset
from app.models.report import Report
from app.schemas.report import ReportRequest, ReportResponse
from app.modules.reports.report_service import ReportService
from app.config import settings

router = APIRouter(tags=["reports"])


@router.post("/datasets/{dataset_id}/reports", response_model=ReportResponse, status_code=202)
def generate_report(
    dataset_id: int,
    request: ReportRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Submits a report generation job (PDF or Excel).
    Runs asynchronously in the background.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if request.report_type not in ["pdf", "excel"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid report type. Allowed values: 'pdf', 'excel'",
        )

    # Create Report record in DB
    report = Report(
        dataset_id=dataset_id,
        report_type=request.report_type,
        status="queued",
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    # Submit to BackgroundTasks
    background_tasks.add_task(ReportService.generate_report_sync, db, report.id)

    return report


@router.get("/reports/{report_id}", response_model=ReportResponse)
def get_report_status(report_id: int, db: Session = Depends(get_db)):
    """Retrieves the generation status of a report."""
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.get("/reports/{report_id}/download")
def download_report(report_id: int, db: Session = Depends(get_db)):
    """Downloads the compiled PDF or Excel report file."""
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    if report.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Report is not ready. Current status: {report.status}",
        )

    absolute_path = settings.data_path / report.storage_path
    if not absolute_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Report file is missing from local storage.",
        )

    filename = f"autoinsight_report_{report.id}.pdf" if report.report_type == "pdf" else f"autoinsight_report_{report.id}.xlsx"
    media_type = "application/pdf" if report.report_type == "pdf" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    return FileResponse(
        path=str(absolute_path),
        filename=filename,
        media_type=media_type,
    )
