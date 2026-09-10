# Aggregated Label Studio ML service
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=9090 \
    WORKERS=1 \
    THREADS=4

RUN apt-get update \
    && apt-get install --no-install-recommends -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/ml-backend-requirements.txt
COPY label_studio_ml/aggregate_backend/requirements.txt /tmp/model-requirements.txt
RUN pip install --no-cache-dir -r /tmp/ml-backend-requirements.txt \
    && pip install --no-cache-dir gunicorn==22.0.0 \
    && pip install --no-cache-dir -r /tmp/model-requirements.txt

COPY . /opt/label-studio-ml-backend
RUN pip install --no-cache-dir --no-deps -e /opt/label-studio-ml-backend

CMD gunicorn --chdir /opt/label-studio-ml-backend \
    --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 \
    label_studio_ml.aggregate_backend._wsgi:app
