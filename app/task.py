# Importations nécessaires
import multiprocessing as mp
import time
from app.celery_app import celery_app

import torch
import json
import gc
from celery import shared_task
from celery.signals import worker_process_init
import os
from whisperx import align,assign_word_speakers, load_audio, load_model, load_align_model, DiarizationPipeline



model, align_model, metadata, diarize_model = None, None, None, None
def init():
    try:
       mp.set_start_method('spawn', force=True)
       print("spawned")
    except RuntimeError:
        print('not spawned')
    pass
    global model, align_model, metadata, diarize_model
    def import_models():
        from app.model import model, align_model, metadata, diarize_model
        return model, align_model, metadata, diarize_model

    model, align_model, metadata, diarize_model = import_models()

from celery.signals import celeryd_init


@celeryd_init.connect
def configure_workers(sender=None, conf=None, **kwargs):
    init()

device = "cuda"
batch_size = 8
compute_type = "float16"
output_file = "transcription_result.json"

def transcribe_audio(audio, batch_size,langue):
    """Transcrit l'audio avec le modèle Whisper préchargé."""
    global model
    with torch.no_grad():
        return model.transcribe(audio, batch_size=batch_size, language=langue)

def align_segments(segments, audio):
    """Aligne les segments avec le modèle d'alignement."""
    return align(segments, align_model, metadata, audio, device, return_char_alignments=False)

def assign_speaker_labels(audio, aligned_result):
    """Assigne les étiquettes de locuteur au résultat."""
    diarize_segments = diarize_model(audio)
    return assign_word_speakers(diarize_segments, aligned_result)

def clear_gpu_cache():
    """Libère la mémoire GPU."""
    gc.collect()
    torch.cuda.empty_cache()

def save_to_json(result, output_path):
    """Sauvegarde le résultat de transcription en JSON."""
    with open(output_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"Transcription sauvegardée dans {output_path}")

import pandas as pd
def merge(df):
    df = df.bfill()
    new_rows = []
    prev_speaker = None
    start = None
    end = None
    text = ""

    for index, row in df.iterrows():
        if prev_speaker is None:
            prev_speaker = row['speaker']
            start = row['start']
            end = row['end']
            text = row['word']
        elif prev_speaker == row['speaker']:
            end = row['end']
            text += " " + row['word']
        else:
            new_rows.append({'start': start, 'end': end, 'word': text, 'speaker': prev_speaker})
            prev_speaker = row['speaker']
            start = row['start']
            end = row['end']
            text = row['word']

    # Append the last row
    if prev_speaker is not None:
        new_rows.append({'start': start, 'end': end, 'word': text, 'speaker': prev_speaker})

    new_df = pd.DataFrame(new_rows)
    return new_rows

def getConversations(data):
    words = []
    for row in data:
        # print(row[])
        words.extend(row['words'])
        # break
    df = pd.DataFrame(words)
    return merge(df)

@shared_task(bind=True)
def run_transcription_task(self, audio_file,langue):
    global model, align_model, metadata, diarize_model

    audio = load_audio(audio_file)
    try:
        # Étape 1 : Transcription
        self.update_state(state='PROGRESS', meta={'status': 'Transcription en cours'})
        print(langue)
        result = transcribe_audio(audio, batch_size,langue)

        # Étape 2 : Alignement
        self.update_state(state='PROGRESS', meta={'status': 'Alignement des segments'})
        result = align_segments(result["segments"], audio)

        # Étape 3 : Attribution des étiquettes de locuteur
        self.update_state(state='PROGRESS', meta={'status': 'Attribution des étiquettes de locuteur'})
        result = assign_speaker_labels(audio, result)

        # Sauvegarde du résultat
        self.update_state(state='PROGRESS', meta={'status': 'Sauvegarde en JSON'})
        conversation = getConversations(result["segments"])
        save_to_json(conversation, output_file)

        self.update_state(state='FINISHED', meta={'status': 'Fin de traitement'})
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


