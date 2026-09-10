# Aggregated ML backend

This first release routes Label Studio projects by their labeling configuration
and supports brush-mask semantic segmentation and OCR.

Template implementations live in `templates/`; third-party API adapters live in
`providers/`. The repository-root `Dockerfile` starts this package through
`label_studio_ml.aggregate_backend._wsgi:app`.

Copy `.env.example` to `.env`, fill in the provider credentials and Label
Studio media path, then run `docker compose up -d --build` from the repository
root. Credentials supplied by a Label Studio model connection take precedence
over the environment fallback values.
