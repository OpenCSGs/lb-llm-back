FROM python:3.12-alpine

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SAM_SERVICE_HOST=0.0.0.0 \
    SAM_SERVICE_PORT=9091 \
    SAM_MODEL_PATH=/app/ml/sam_vit_b_01ec64.pth

WORKDIR /app

RUN addgroup -S sam && adduser -S -G sam sam

COPY --chown=sam:sam sam_mock_backend.py /app/sam_mock_backend.py
COPY --chown=sam:sam ml /app/ml

USER sam

EXPOSE 9091

HEALTHCHECK --interval=10s --timeout=3s --start-period=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9091/health', timeout=2)" || exit 1

CMD ["python", "sam_mock_backend.py"]
