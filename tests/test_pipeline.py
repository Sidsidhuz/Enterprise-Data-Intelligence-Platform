from __future__ import annotations

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db
from app.config import settings, Settings

# Define a temporary testing database URL
TEST_DATABASE_URL = "sqlite:///./test_autoinsight.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    # Setup test database tables
    Base.metadata.create_all(bind=engine)
    
    # Temporarily override settings database URL & ensure paths exist
    old_db_url = settings.database_url
    settings.database_url = TEST_DATABASE_URL
    settings.ensure_data_dirs_exist()
    
    yield
    
    # Dispose of engines to release connection handles
    from app.database import engine as app_engine
    app_engine.dispose()
    engine.dispose()
    
    # Tear down database tables
    try:
        Base.metadata.drop_all(bind=engine)
    except Exception:
        pass
        
    if os.path.exists("./test_autoinsight.db"):
        try:
            os.remove("./test_autoinsight.db")
        except PermissionError:
            pass
    
    # Clean up test directories
    # (Optional, since we write files relative to PROJECT_ROOT/data)
    settings.database_url = old_db_url


# Override FastAPI DB dependency
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_and_pipeline_flow(client):
    # 1. Prepare dummy CSV dataset
    csv_content = (
        "age,income,region,has_house,target_default\n"
        "25.0,50000.0,east,1,0\n"
        "30.0,60000.5,west,0,0\n"
        "45.0,120000.0,east,1,1\n"
        "21.0,30000.0,north,0,0\n"
        "35.0,80000.0,south,1,1\n"
        "28.0,45000.0,,0,0\n"  # row with missing region
        "30.0,60000.5,west,0,0\n" # duplicate row
    )
    
    # Create upload file
    files = {"file": ("test_loans.csv", csv_content, "text/csv")}
    
    # 2. Upload dataset
    response = client.post("/api/v1/datasets", files=files)
    assert response.status_code == 201
    dataset_data = response.json()
    assert dataset_data["filename"] == "test_loans.csv"
    assert dataset_data["status"] == "profiled"
    dataset_id = dataset_data["id"]

    # 3. Check profile metadata
    response = client.get(f"/api/v1/datasets/{dataset_id}/profile")
    assert response.status_code == 200
    profile_data = response.json()
    assert profile_data["row_count"] == 7
    assert profile_data["column_count"] == 5
    assert profile_data["dtypes"]["age"] == "numeric"
    assert profile_data["dtypes"]["region"] == "categorical"
    assert profile_data["duplicates"] == 1  # 1 duplicate row

    # 4. Clean dataset
    cleaning_payload = {"region": "mode", "income": "median"}
    response = client.post(f"/api/v1/datasets/{dataset_id}/clean", json=cleaning_payload)
    assert response.status_code == 200
    assert response.json()["status"] == "cleaned"

    # 5. Check EDA calculations
    response = client.get(f"/api/v1/datasets/{dataset_id}/eda")
    assert response.status_code == 200
    eda = response.json()
    assert "correlation" in eda
    assert "distributions" in eda
    assert len(eda["correlation"]["columns"]) > 0

    # 6. Train AutoML models
    training_payload = {
        "target_column": "target_default",
        "problem_type": "classification",
        "algorithms": ["random_forest", "logistic_regression"],
        "tuning_budget_seconds": 30,
    }
    response = client.post(f"/api/v1/datasets/{dataset_id}/train", json=training_payload)
    assert response.status_code == 202
    job_data = response.json()
    job_id = job_data["id"]
    assert job_data["status"] in ["queued", "running", "completed"]

    # Wait synchronously for training to finish (background task)
    import time
    max_wait = 10
    while max_wait > 0:
        response = client.get(f"/api/v1/training-jobs/{job_id}")
        job_status = response.json()
        if job_status["status"] in ["completed", "failed"]:
            break
        time.sleep(1)
        max_wait -= 1

    assert job_status["status"] == "completed"
    assert len(job_status["leaderboard"]) > 0
    best_model_id = job_status["leaderboard"][0]["model_id"]

    # 7. Model serving: Single prediction
    predict_payload = {
        "input_data": {
            "age": 32,
            "income": 70000.0,
            "region": "west",
            "has_house": 1,
        }
    }
    response = client.post(f"/api/v1/models/{best_model_id}/predict", json=predict_payload)
    assert response.status_code == 200
    pred_res = response.json()
    assert "prediction" in pred_res
    assert "probability" in pred_res
    assert "explanation" in pred_res
    assert len(pred_res["explanation"]["shap_values"]) > 0

    # 8. Model serving: Batch predictions
    batch_csv_content = (
        "age,income,region,has_house\n"
        "26,55000.0,east,0\n"
        "40,110000.0,west,1\n"
    )
    batch_files = {"file": ("batch_test.csv", batch_csv_content, "text/csv")}
    response = client.post(f"/api/v1/models/{best_model_id}/predict-batch", files=batch_files)
    assert response.status_code == 200
    csv_response = response.text
    assert "prediction" in csv_response
    assert "probability" in csv_response

    # 9. Report generation: PDF report
    response = client.post(f"/api/v1/datasets/{dataset_id}/reports", json={"report_type": "pdf"})
    assert response.status_code == 202
    report_id = response.json()["id"]

    # Wait for report generation to finish
    max_wait = 10
    while max_wait > 0:
        response = client.get(f"/api/v1/reports/{report_id}")
        if response.json()["status"] in ["completed", "failed"]:
            break
        time.sleep(1)
        max_wait -= 1

    assert response.json()["status"] == "completed"

    # Download PDF
    response = client.get(f"/api/v1/reports/{report_id}/download")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
