import gc;
import os
import shutil

import torch
from celery.result import AsyncResult
from fastapi import FastAPI, UploadFile, File, HTTPException, Query

gc.collect(); torch.cuda.empty_cache();
from pathlib import Path
from datetime import datetime
import multiprocessing as mp

from app.task import run_transcription_task

try:
   mp.set_start_method('spawn', force=True)
   print("spawned")
except RuntimeError:
   pass

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

from uuid import uuid1


@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...), langue: str = None):
    try:
        # Generate unique file path
        save_path = os.path.join(UPLOAD_DIR, f"{str(uuid1())}_{file.filename}")

        # Save the uploaded file
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"File saved to path: {save_path}")

        # Trigger transcription task
        task = run_transcription_task.delay(save_path, langue)

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
