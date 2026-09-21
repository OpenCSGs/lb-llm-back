import base64
import json
import logging
import mimetypes
import os

from json_repair import repair_json
import requests
from PIL import Image

from ..providers.errors import raise_for_status_with_detail

logger = logging.getLogger(__name__)


class DoubaoDocument:
    """Extract labeled regions from document pages in bounded Seed requests."""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('ARK_API_KEY')
        self.model = os.getenv('DOUBAO_MODEL', 'doubao-seed-2-1-turbo-260628')
        self.base_url = os.getenv(
            'ARK_BASE_URL', 'https://ark.cn-beijing.volces.com/api/v3'
        ).rstrip('/')
        self.timeout = int(os.getenv('DOUBAO_TIMEOUT', '360'))
        self.batch_size = max(1, int(os.getenv('DOUBAO_DOCUMENT_BATCH_SIZE', '10')))

    @staticmethod
    def _parse(answer):
        answer = answer.strip()
        if answer.startswith('```'):
            answer = answer.strip('`').removeprefix('json').strip()
        try:
            return json.loads(answer)
        except json.JSONDecodeError:
            parsed = repair_json(answer, return_objects=True)
            if not isinstance(parsed, dict):
                raise
            return parsed

    @staticmethod
    def _prompt(labels):
        prompt = (
            '你是多页文档版面标注模型。按输入图片顺序从0开始编号页面，找出属于项目标签的'
            '所有清晰区域。标签必须且只能从以下列表选择：'
            f'{json.dumps(labels, ensure_ascii=False)}。'
            '每个区域返回紧贴内容的矩形bbox，坐标按该页原图宽高归一化到0到1000，格式为'
            '[left,top,right,bottom]。不要返回不属于标签的区域，不得创造标签。严格只输出json对象：'
            '{"regions":[{"page_index":0,"label":"标签","bbox":[x1,y1,x2,y2],'
            '"confidence":0.95}]}。'
        )
        return prompt

    def _extract_batch(self, page_paths, labels, page_offset, batch_number, batch_count):
        prompt = self._prompt(labels)
        content = [{'type': 'text', 'text': prompt}]
        page_sizes = []
        for path in page_paths:
            mime = mimetypes.guess_type(path)[0] or 'application/octet-stream'
            with open(path, 'rb') as file:
                raw = file.read()
            with Image.open(path) as image:
                page_sizes.append(image.size)
            url = f'data:{mime};base64,{base64.b64encode(raw).decode("ascii")}'
            content.append({'type': 'image_url', 'image_url': {'url': url, 'detail': 'high'}})
        logger.info(
            'Doubao document request: model=%s batch=%d/%d page_offset=%d pages=%d '
            'labels=%s sizes=%s',
            self.model, batch_number, batch_count, page_offset, len(page_paths),
            labels, page_sizes,
        )
        response = requests.post(
            f'{self.base_url}/chat/completions',
            headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'},
            json={
                'model': self.model,
                'messages': [{'role': 'user', 'content': content}],
                'temperature': 0,
                'thinking': {'type': 'disabled'},
                'response_format': {'type': 'json_object'},
                'max_tokens': 8192,
            },
            timeout=self.timeout,
        )
        if not response.ok:
            logger.error(
                'Doubao document request failed: status=%s body=%s',
                response.status_code,
                response.text,
            )
        raise_for_status_with_detail(response)
        answer = response.json()['choices'][0]['message']['content']
        logger.info(
            'Doubao document raw response: batch=%d/%d page_offset=%d response=%s',
            batch_number, batch_count, page_offset, answer,
        )
        parsed = self._parse(answer)
        regions = []
        for raw in parsed.get('regions', []):
            if not isinstance(raw, dict) or raw.get('label') not in labels:
                continue
            try:
                page_index = int(raw['page_index'])
                bbox = [float(value) for value in raw['bbox'][:4]]
            except (KeyError, TypeError, ValueError):
                continue
            if page_index < 0 or page_index >= len(page_sizes) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [min(1000.0, max(0.0, value)) for value in bbox]
            if x2 <= x1 or y2 <= y1:
                continue
            regions.append({
                'page_index': page_offset + page_index, 'label': raw['label'],
                'x': x1 / 10, 'y': y1 / 10,
                'width': (x2 - x1) / 10, 'height': (y2 - y1) / 10,
                'confidence': min(1.0, max(0.0, float(raw.get('confidence', 1.0)))),
                'original_width': page_sizes[page_index][0],
                'original_height': page_sizes[page_index][1],
            })
        return regions

    def extract(self, page_paths, labels):
        if not self.api_key:
            raise ValueError('ARK_API_KEY is required for Doubao document analysis')
        if not page_paths:
            raise ValueError('At least one document page is required')
        if not labels:
            raise ValueError('At least one document label is required')

        batch_count = (len(page_paths) + self.batch_size - 1) // self.batch_size
        logger.info(
            'Doubao document batching: total_pages=%d batch_size=%d batches=%d',
            len(page_paths), self.batch_size, batch_count,
        )
        regions = []
        for batch_index, page_offset in enumerate(
            range(0, len(page_paths), self.batch_size), start=1
        ):
            batch_paths = page_paths[page_offset:page_offset + self.batch_size]
            regions.extend(
                self._extract_batch(
                    batch_paths, labels, page_offset, batch_index, batch_count
                )
            )
        logger.info(
            'Doubao document merged result: total_pages=%d batches=%d regions=%s',
            len(page_paths), batch_count, regions,
        )
        return regions
