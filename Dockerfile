FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy only the deployment code (and artifacts) into the image
COPY deployment ./deployment

# Install system dependencies (needed for some Python packages like xgboost)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for deployment
RUN pip install --no-cache-dir -r deployment/requirements.txt

# Switch into the deployment folder
WORKDIR /app/deployment

# Expose Gradio's default port
EXPOSE 7860

# Command to run the Gradio app
CMD ["python", "gradio_app.py"]
