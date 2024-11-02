# app/model_loader.py
import whisperx

device = "cuda"
batch_size = 8
compute_type = "float16"
model, align_model, metadata, diarize_model = None, None, None, None

def initialize_models():
    global model, align_model, metadata, diarize_model
    if model is None:
        print("Loading models at startup...")
        model = whisperx.load_model("large-v3", device, compute_type=compute_type)
        align_model, metadata = whisperx.load_align_model(language_code='fr', device=device)
        diarize_model = whisperx.DiarizationPipeline(device=device)
        print("Models loaded successfully.")
