# Label Studio Magic Wand ML Gate

This directory contains the Label Studio source files changed for the Magic
Wand ML-service gate. Paths below this directory mirror their paths in the
Label Studio repository and can be copied back to the same locations.

Behavior:

- Magic Wand clicks call `POST /api/ml/magic-wand` before local rendering.
- The backend waits for the configured ML service and returns its payload in
  `ml_result` for inspection.
- The editor deliberately ignores `ml_result` and renders the original local
  Magic Wand result.
- When the feature is disabled, the editor displays an internationalized
  system toast asking the user to contact an administrator.

Configuration is documented in `.env.example`.

The included `docker-compose.yml` passes these variables to the Label Studio
`app` container. Its default ML URL is
`http://host.docker.internal:9091/predict`, which reaches a SAM test service
published on the Docker host. Override `MAGIC_WAND_ML_URL` when both services
share a Docker network, for example `http://sam-service:9091/predict`.
