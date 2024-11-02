# Importations nécessaires
import multiprocessing as mp
import time
from app.celery_app import celery_app
import whisperx
import torch
import json
import gc
from celery import shared_task
from celery.signals import worker_process_init
import os
# Configuration
device = "cuda"
batch_size = 8
compute_type = "float16"
output_file = "transcription_result.json"

model, align_model, metadata, diarize_model = None, None, None, None

@worker_process_init.connect
def initialize_models(**kwargs):
    """Initialise les modèles lors du démarrage de chaque worker."""
    global model, align_model, metadata, diarize_model
    if model is None:
        print("Chargement des modèles au démarrage du worker...")
        model = whisperx.load_model("large-v3", device, compute_type=compute_type)
        align_model, metadata = whisperx.load_align_model(language_code='fr', device=device)
        diarize_model = whisperx.DiarizationPipeline(device=device)
        print("Modèles chargés avec succès.")

def transcribe_audio(audio, batch_size):
    """Transcrit l'audio avec le modèle Whisper préchargé."""
    with torch.no_grad():
        return model.transcribe(audio, batch_size=batch_size, language='fr')

def align_segments(segments, audio):
    """Aligne les segments avec le modèle d'alignement."""
    return whisperx.align(segments, align_model, metadata, audio, device, return_char_alignments=False)

def assign_speaker_labels(audio, aligned_result):
    """Assigne les étiquettes de locuteur au résultat."""
    diarize_segments = diarize_model(audio)
    return whisperx.assign_word_speakers(diarize_segments, aligned_result)

def clear_gpu_cache():
    """Libère la mémoire GPU."""
    gc.collect()
    torch.cuda.empty_cache()

def save_to_json(result, output_path):
    """Sauvegarde le résultat de transcription en JSON."""
    with open(output_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"Transcription sauvegardée dans {output_path}")

@shared_task(bind=True)
def run_transcription_task(self, audio_file):
    global model, align_model, metadata, diarize_model
    if model is None or align_model is None or diarize_model is None:
        initialize_models()  # Force l'initialisation si les modèles ne sont pas chargés

    audio = whisperx.load_audio(audio_file)
    try:
        # Étape 1 : Transcription
        self.update_state(state='PROGRESS', meta={'status': 'Transcription en cours'})
        result = transcribe_audio(audio, batch_size)

        # Étape 2 : Alignement
        self.update_state(state='PROGRESS', meta={'status': 'Alignement des segments'})
        result = align_segments(result["segments"], audio)

        # Étape 3 : Attribution des étiquettes de locuteur
        self.update_state(state='PROGRESS', meta={'status': 'Attribution des étiquettes de locuteur'})
        result = assign_speaker_labels(audio, result)

        # Sauvegarde du résultat
        self.update_state(state='PROGRESS', meta={'status': 'Sauvegarde en JSON'})
        save_to_json(result["segments"], output_file)

    except torch.cuda.OutOfMemoryError:
        print("Erreur : mémoire GPU insuffisante.")
        raise Exception("La mémoire GPU est insuffisante pour le traitement.")

    except Exception as e:
        print(f"Une erreur est survenue : {e}")
        raise e

    finally:
        # Libère la mémoire et supprime les données
        clear_gpu_cache()
        gc.collect()
        if os.path.exists(audio_file):
            os.remove(audio_file)

    return f"Transcription sauvegardée dans {output_file}"


