# Aggregated Label Studio ML service
ARG PYTHON_IMAGE=m.daocloud.io/docker.io/library/python:3.12-slim
FROM ${PYTHON_IMAGE}

ARG APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ARG PIP_DEFAULT_TIMEOUT=180
ARG PIP_RETRIES=10

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=9090 \
    WORKERS=1 \
    THREADS=4 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} \
    PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT} \
    PIP_RETRIES=${PIP_RETRIES}

RUN sed -i \
        -e "s|http://deb.debian.org|${APT_MIRROR}|g" \
        -e "s|http://security.debian.org|${APT_MIRROR}|g" \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
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
