import base64
import io
import json
import logging
import os
from typing import List

import numpy as np
import requests
from ..providers.errors import raise_for_status_with_detail
from PIL import Image

logger = logging.getLogger(__name__)


class DoubaoMaskClassifier:
    """Assign one Label Studio label to each EntitySegment mask."""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("ARK_API_KEY")
        self.model = os.getenv("DOUBAO_MODEL", "doubao-seed-2-1-turbo-260628")
        self.base_url = os.getenv(
            "ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"
        ).rstrip("/")
        self.timeout = int(os.getenv("DOUBAO_TIMEOUT", "60"))
        self.max_image_size = int(os.getenv("DOUBAO_MAX_IMAGE_SIZE", "640"))

    def _data_url(self, image: Image.Image) -> str:
        image = image.convert("RGB")
        image.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=80, optimize=True)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _highlight(image: Image.Image, mask: np.ndarray) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        selected = mask.astype(bool)
        highlighted = (rgb.astype(np.float32) * 0.25).astype(np.uint8)
        target = rgb[selected].astype(np.float32)
        target[:, 0] = np.clip(target[:, 0] * 0.45 + 140, 0, 255)
        target[:, 1:] *= 0.55
        highlighted[selected] = target.astype(np.uint8)
        return Image.fromarray(highlighted, mode="RGB")

    def classify(self, image_path: str, masks: List[np.ndarray], labels: List[str]) -> List[str]:
        if not self.api_key:
            raise ValueError("ARK_API_KEY is required for Doubao mask classification")
        if not labels:
            raise ValueError("Label Studio BrushLabels must contain at least one Label")
        if len(labels) == 1:
            return [labels[0]] * len(masks)

        image = Image.open(image_path).convert("RGB")
        original_url = self._data_url(image)
        allowed = json.dumps(labels, ensure_ascii=False)
        content = [{
            "type": "text",
            "text": (
                "你是图像语义分割标注器。第一张是原图，后续图片依次是 Mask 1、Mask 2……，"
                "每张 Mask 图中待分类区域以红色高亮，其他区域已变暗。请为每个 Mask 只从"
                f"以下标签中选择一个：{allowed}。严格输出 JSON 数组，顺序与 Mask 图片一致，"
                "例如：[\"标签1\", \"标签2\"]，不要解释。"
            ),
        }, {
            "type": "image_url",
            "image_url": {"url": original_url},
        }]

        for index, mask in enumerate(masks, 1):
            highlighted_url = self._data_url(self._highlight(image, mask))
            content.extend([
                {"type": "text", "text": f"Mask {index}"},
                {"type": "image_url", "image_url": {"url": highlighted_url}},
            ])

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "thinking": {"type": "disabled"},
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        raise_for_status_with_detail(response)
        answer = response.json()["choices"][0]["message"]["content"].strip()
        if answer.startswith("```"):
            answer = answer.strip("`").removeprefix("json").strip()
        classified = json.loads(answer)
        if not isinstance(classified, list) or len(classified) != len(masks):
            raise RuntimeError(
                f"Doubao returned {classified!r}, expected {len(masks)} labels"
            )
        if any(label not in labels for label in classified):
            raise RuntimeError(
                f"Doubao returned labels {classified!r}, expected only {labels!r}"
            )
        logger.info("Doubao classified masks as %s", classified)

        return classified
