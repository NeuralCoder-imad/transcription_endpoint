# Importations nécessaires
import multiprocessing as mp
from time import time
from app.celery_app import celery_app
import torch
import json
import gc
from celery import shared_task
from celery.signals import worker_process_init
import os

from speechbrain.utils.edit_distance import alignment

#from pyannote.audio.models.embedding.wespeaker.convert import duration

from whisperx import align,assign_word_speakers, load_audio, load_model, load_align_model, DiarizationPipeline
from datetime import datetime


model, align_model, metadata, diarize_model = None, None, None, None
from celery.signals import worker_process_init


def init(*args, **kwargs):
    try:
        mp.set_start_method('spawn', force=True)
        print("spawned")
    except RuntimeError:
        print('not spawned')
    global model, align_model, metadata, diarize_model
    # Initialisation directe des modèles sans fonction imbriquée
    from app.model import model, align_model, metadata, diarize_model

from celery.signals import celeryd_init,worker_shutdown


@celeryd_init.connect
def configure_workers(sender=None, conf=None, **kwargs):
    init()

device = "cuda"
batch_size = 8
compute_type = "float16"
output_file = "transcription_result.json"
from threading import Thread


def transcribe_audio(audio, batch_size,langue):
    """Transcrit l'audio avec le modèle Whisper préchargé."""
    global model
    with torch.no_grad():
        return model.transcribe(audio, batch_size=batch_size, language=langue)

def align_segments(segments, audio):
    """Aligne les segments avec le modèle d'alignement."""
    return align(segments, align_model, metadata, audio, device, return_char_alignments=False)

def assign_speaker_labels(audio, aligned_result,num_speakers, min_speakers, max_speakers):
    """Assigne les étiquettes de locuteur au résultat."""
    print(num_speakers)
    diarize_segments = diarize_model(audio, num_speakers=None, min_speakers=None, max_speakers=None)
    return assign_word_speakers(diarize_segments, aligned_result)

def clear_gpu_cache():
    """Libère la mémoire GPU."""
    gc.collect()
    torch.cuda.empty_cache()

def save_to_json(result, output_path):
    #%%
    with open('resultat.txt', 'w', encoding='utf-8') as f:
        for discussion in result:
            text =f"{discussion['speaker']}: {discussion['text']}\n"
            f.write(text)
    print(f"Transcription sauvegardée ")

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

from threading import Lock

transcribe_locker = Lock()
alignment_locker=Lock()
diarize_locker=Lock()
save_locker=Lock()

@shared_task(bind=True)
def run_transcription_task(self, audio_file,langue,num_speakers, min_speakers, max_speakers):
    global model, align_model, metadata, diarize_model
    start = time()
    audio = load_audio(audio_file)
    try:
        # Étape 1 : Transcription
        with transcribe_locker:

            self.update_state(state='PROGRESS', meta={'status': 'Transcription en cours'})
            result = transcribe_audio(audio, batch_size,langue)

        # Étape 2 : Alignement
        with alignment_locker:

            self.update_state(state='PROGRESS', meta={'status': 'Alignement des segments'})
            result = align_segments(result["segments"], audio)

        # Étape 3 : Attribution des étiquettes de locuteur
        with diarize_locker:
            self.update_state(state='PROGRESS', meta={'status': 'Attribution des étiquettes de locuteur'})
            result = assign_speaker_labels(audio, result,num_speakers, min_speakers, max_speakers)

        # Sauvegarde du résultat
        with save_locker:
                self.update_state(state='PROGRESS', meta={'status': 'Sauvegarde en JSON'})
              #  conversation = getConversations(©©ÂÂ)
                save_to_json(result["segments"], output_file)
                end = time()
                self.update_state(state='FINISHED', meta={'status': 'Fin de traitement','start':start,'end':end})
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
    end = time()
    duration=end-start
    return {"statut":"transcripted ","start":start,"end":end,"duration":duration}








