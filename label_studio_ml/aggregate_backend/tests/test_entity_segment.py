import base64
import io

import numpy as np
from PIL import Image

from label_studio_ml.aggregate_backend.providers.entity_segment import (
    VolcengineEntitySegment,
)


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.payload = None

    def cv_process(self, payload):
        self.payload = payload
        return self.response


def encode_mask(mask):
    output = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8)).save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def test_entity_segment_decodes_masks(monkeypatch):
    monkeypatch.setenv("VOLCENGINE_ENTITY_SEGMENT_MAX_ENTITY", "2")
    encoded = encode_mask(np.array([[0, 255], [255, 0]]))
    client = FakeClient({
        "code": 10000,
        "message": "Success",
        "data": {"binary_data_base64": [encoded]},
    })

    result = VolcengineEntitySegment(client=client).segment_bytes(b"image")

    assert result["masks"][0].tolist() == [[0, 1], [1, 0]]
    assert result["probs"] == [1.0]
    assert client.payload["req_key"] == "entity_seg"
    assert client.payload["max_entity"] == 2


def test_entity_segment_skips_leading_original_image():
    original = encode_mask(np.array([[10, 20], [30, 40]]))
    encoded_mask = encode_mask(np.array([[0, 255], [255, 0]]))
    client = FakeClient({
        "code": 10000,
        "message": "Success",
        "data": {
            "binary_data_base64": [original, encoded_mask],
            "entity_num": [1],
        },
    })

    result = VolcengineEntitySegment(client=client).segment_bytes(b"image")

    assert len(result["masks"]) == 1
    assert result["masks"][0].tolist() == [[0, 1], [1, 0]]


def test_entity_segment_raises_api_error():
    client = FakeClient({"code": 50400, "message": "Unauthorized"})

    try:
        VolcengineEntitySegment(client=client).segment_bytes(b"image")
    except RuntimeError as exc:
        assert "50400" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")
