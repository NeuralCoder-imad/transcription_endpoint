# Use a Python base image with CUDA support
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Install Python, Git, and additional dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.9 \
    python3-pip \
    curl \
    gnupg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libcudnn8 \
    libcudnn8-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone WhisperX repository and install it as editable
RUN git clone https://github.com/m-bain/whisperX.git && \
    cd whisperX && \
    pip install -e .

RUN apt-get update && apt-get install -y ffmpeg

# Set the working directory
WORKDIR /app

# Copy application files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000


# Copier le reste de l’application
COPY . .

# Exposer les ports pour FastAPI et Flower
EXPOSE 5000 5555

# Lancer FastAPI, le worker Celery et Flower
CMD ["sh", "-c", "CUDA_VISIBLE_DEVICES=0 celery -A app.task worker --loglevel info --pool=threads & uvicorn main:app --host 0.0.0.0 --port 5000 "]
#CMD ["sh", "-c", "celery -A celery_client worker -l info -P solo & uvicorn app:app --host 0.0.0.0 --port 5000"]
