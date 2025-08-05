FROM python:3.8-slim-buster


ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.txt /app/requirements.txt


RUN apt update -y && \
    apt install -y --no-install-recommends awscli build-essential gcc && \
    pip install --no-cache-dir -r /app/requirements.txt


# Copy the rest of the application code
COPY . /app

CMD ["python3", "app.py"]

