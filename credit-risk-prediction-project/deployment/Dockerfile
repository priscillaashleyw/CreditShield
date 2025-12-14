FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT="7860"

WORKDIR /app

# System deps (xgboost often needs libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir huggingface_hub

COPY . .

# Download artefacts from a PUBLIC HF model repo into /app/model_artifacts
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='wangyixuan818/creditshield-artifacts', local_dir='model_artifacts', local_dir_use_symlinks=False)"

EXPOSE 7860
CMD ["python", "app.py"]
