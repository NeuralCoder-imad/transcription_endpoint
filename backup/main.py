from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from celery.result import AsyncResult
import whisperx
import shutil
import os
import tempfile
from app.task import run_transcription_task
from pathlib import Path
import multiprocessing
from datetime import datetime

import multiprocessing as mp

try:
   mp.set_start_method('spawn', force=True)
   print("spawned")
except RuntimeError:
   pass

app = FastAPI()





UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"File saved to path: {file_path}")

        # Trigger transcription task
        task = run_transcription_task.delay(str(file_path))

        # Return the task ID for tracking
        return {"task_id": task.id, "message": "Transcription started"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    Récupère le statut actuel de la tâche, y compris l'étape et d'autres métadonnées pertinentes.
    """
    task_result = AsyncResult(task_id)

    # Adding metadata for the task response
    task_info = {
        "task_id": task_id,
        "status": task_result.state,
        "result": task_result.result if task_result.state == "SUCCESS" else None,  # Include result if task is completed
        "date_created": datetime.utcnow().isoformat(),  # Metadata example: current timestamp in UTC
        "meta": task_result.info if task_result.state == "PROGRESS" else None  # Custom metadata for progress tracking
    }

    return task_info
