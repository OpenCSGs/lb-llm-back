import base64
import io
import os
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


class VolcengineEntitySegment:
    """Small adapter around Volcengine's EntitySegment API."""

    SUCCESS_CODE = 10000

    def __init__(self, client=None, access_key=None, secret_key=None):
        self._client = client
        self.access_key = access_key or os.getenv("VOLC_ACCESSKEY")
        self.secret_key = secret_key or os.getenv("VOLC_SECRETKEY")
        self.model = os.getenv("VOLCENGINE_ENTITY_SEGMENT_MODEL", "entity_seg")
        self.max_entity = int(os.getenv("VOLCENGINE_ENTITY_SEGMENT_MAX_ENTITY", "10"))
        self.refine_mask = int(os.getenv("VOLCENGINE_ENTITY_SEGMENT_REFINE_MASK", "1"))
        self.return_format = int(os.getenv("VOLCENGINE_ENTITY_SEGMENT_RETURN_FORMAT", "3"))

    def _get_client(self):
        if self._client is None:
            from volcengine.visual.VisualService import VisualService

            access_key = self.access_key
            secret_key = self.secret_key
            if not access_key or not secret_key:
                raise ValueError(
                    "VOLC_ACCESSKEY and VOLC_SECRETKEY are required when "
                    "USE_THIRD_PARTY_MODELS=true"
                )
            self._client = VisualService()
            self._client.set_ak(access_key)
            self._client.set_sk(secret_key)
        return self._client

    @staticmethod
    def _decode_mask(encoded: str) -> np.ndarray:
        raw = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(raw))

        # EntitySegment may return either an alpha mask, a grayscale mask, or
        # a foreground cutout. Prefer alpha when it contains useful data.
        if "A" in image.getbands():
            alpha = np.asarray(image.getchannel("A"))
            if np.any(alpha < 255):
                return (alpha > 0).astype(np.uint8)

        grayscale = np.asarray(image.convert("L"))
        # EntitySegment returns soft grayscale masks. Values close to zero are
        # edge confidence/noise, not foreground; threshold at the midpoint
        # before converting to Label Studio's binary BrushLabels mask.
        return (grayscale >= 128).astype(np.uint8)

    def segment_bytes(self, image_bytes: bytes) -> Dict[str, List]:
        payload = {
            "req_key": self.model,
            "binary_data_base64": [base64.b64encode(image_bytes).decode("ascii")],
            "max_entity": self.max_entity,
            "refine_mask": self.refine_mask,
            "return_format": self.return_format,
        }
        # Intelligent segmentation is routed through the generic CVProcess
        # action; req_key selects the entity_seg algorithm.
        response = self._get_client().cv_process(payload)
        if response.get("code") != self.SUCCESS_CODE:
            raise RuntimeError(
                "Volcengine EntitySegment failed: "
                f"code={response.get('code')}, message={response.get('message')}"
            )

        data = response.get("data", {})
        encoded_masks = data.get("binary_data_base64") or []

        # With return_format=3, CVProcess returns the original RGB image first,
        # followed by one grayscale mask per entity. entity_num describes only
        # the entity masks, so exclude the leading source image.
        entity_num = data.get("entity_num") or []
        entity_count = entity_num[0] if entity_num else None
        if entity_count is not None and len(encoded_masks) == entity_count + 1:
            encoded_masks = encoded_masks[1:]

        masks = [self._decode_mask(item) for item in encoded_masks]
        masks = [mask for mask in masks if np.any(mask)]
        return {"masks": masks, "probs": [1.0] * len(masks)}

    def segment_file(self, image_path: str) -> Dict[str, List]:
        with open(image_path, "rb") as image_file:
            return self.segment_bytes(image_file.read())
