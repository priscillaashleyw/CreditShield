FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (xgboost often needs build tools on slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the deployment folder (nested in your repo)
COPY credit-risk-prediction-project/deployment ./deployment

# Install deps
RUN pip install --no-cache-dir -r /app/deployment/requirements.txt

WORKDIR /app/deployment
EXPOSE 7860

CMD ["python", "gradio_app.py"]
