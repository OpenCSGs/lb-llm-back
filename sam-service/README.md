# SAM Test Backend

This is a lightweight test copy of the neighboring `sam` project. It keeps
the Label Studio ML service endpoints but does not load SAM or perform real
inference. `/predict` returns a random RLE brush mask after an optional delay.
The `ml/` directory mirrors the original project and contains the SAM ViT-B
checkpoint. The test service intentionally does not load the checkpoint.

The Label Studio Magic Wand integration only waits for this response. It never
renders the random prediction; after the response arrives, the editor runs its
existing pixel-based Magic Wand algorithm and renders that real local result.

## Run

```bash
python3 sam_mock_backend.py
```

The default endpoint is `http://localhost:9091/predict`.

## Docker Compose

```bash
docker compose up -d --build
docker compose ps
curl http://localhost:9091/health
```

Stop the service with:

```bash
docker compose down
```

To publish a different host port or adjust the response delay:

```bash
SAM_SERVICE_PORT=19091 SAM_SERVICE_DELAY_SECONDS=1 docker compose up -d --build
```
