FROM python:3.10-slim-bookworm

# Avoid prompts during builds
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies and Python packages
COPY requirements.txt /app/requirements.txt

RUN apt update -y && \
    apt install -y --no-install-recommends awscli build-essential gcc && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy source code
COPY . /app

# Run the app
CMD ["python3", "app.py"]
