from sqlalchemy.log import echo_property

import whisperx
model, align_model, metadata, diarize_model = None, None, None, None
device = "cuda"
batch_size = 8
compute_type = "float16"
output_file = "transcription_result.json"

def initialize_models():
    """Initialise les modèles lors du démarrage de chaque worker."""
    print("Chargement des modèles au démarrage du worker...")
    model = whisperx.load_model("large-v3", device, compute_type=compute_type, num_workers=2)
    align_model, metadata = whisperx.load_align_model(language_code='fr', device=device)
    diarize_model = whisperx.DiarizationPipeline(device=device)
    print("Modèles chargés avec succès.")

    return model, align_model, metadata, diarize_model

if model == None:
    model, align_model, metadata, diarize_model = initialize_models()
