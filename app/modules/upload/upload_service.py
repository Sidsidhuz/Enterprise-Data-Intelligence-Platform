from __future__ import annotations

import os
from pathlib import Path
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.models.dataset import Dataset


class UploadService:
    @staticmethod
    def handle_upload(db: Session, file: UploadFile) -> Dataset:
        """
        Receives an uploaded file, validates it (extension and size),
        creates a database record for it, and saves it to data/raw/{dataset_id}/original.{ext}.
        """
        filename = file.filename
        if not filename:
            raise HTTPException(status_code=400, detail="Filename cannot be empty")

        # Validate file extension
        ext = os.path.splitext(filename)[1].lstrip(".").lower()
        if ext not in settings.allowed_extensions_set:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file extension. Allowed: {settings.allowed_extensions}",
            )

        # Create temporary directory inside project root to read and check size first,
        # or we can write to db first, get ID, and stream directly to final location.
        # Writing to DB first is very clean because we get the unique ID right away!
        dataset = Dataset(
            filename=filename,
            storage_path="pending",  # Will update after saving
            status="uploaded",
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)

        try:
            # Prepare final storage path
            dataset_dir = settings.raw_dir / str(dataset.id)
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Keep original extension or standardise
            file_extension = f".{ext}"
            relative_storage_path = f"raw/{dataset.id}/original{file_extension}"
            absolute_storage_path = settings.data_path / relative_storage_path

            # Write file in chunks to enforce size limit
            total_bytes = 0
            max_bytes = settings.max_upload_size_bytes

            with open(absolute_storage_path, "wb") as f:
                while chunk := file.file.read(1024 * 1024):  # 1MB chunks
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise HTTPException(
                            status_code=400,
                            detail=f"File exceeds maximum allowed size of {settings.max_upload_size_mb}MB.",
                        )
                    f.write(chunk)

            # Update dataset with correct storage path
            dataset.storage_path = relative_storage_path
            db.commit()
            db.refresh(dataset)

        except Exception as e:
            # Clean up database and files if upload failed mid-way
            db.delete(dataset)
            db.commit()
            # Clean directory if exists
            dataset_dir = settings.raw_dir / str(dataset.id)
            if dataset_dir.exists():
                import shutil
                shutil.rmtree(dataset_dir)
                
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Failed to save upload: {str(e)}")

        return dataset
