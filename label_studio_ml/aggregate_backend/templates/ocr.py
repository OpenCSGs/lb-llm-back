import base64
import json
import logging
import mimetypes
import os

import requests
from PIL import Image
from json_repair import repair_json
from ..providers.errors import raise_for_status_with_detail

logger = logging.getLogger(__name__)

class DoubaoOCR:
    """General OCR using Doubao Seed vision with normalized text polygons."""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('ARK_API_KEY')
        self.model = os.getenv('DOUBAO_MODEL', 'doubao-seed-2-1-turbo-260628')
        self.base_url = os.getenv(
            'ARK_BASE_URL', 'https://ark.cn-beijing.volces.com/api/v3'
        ).rstrip('/')
        self.timeout = int(os.getenv('DOUBAO_TIMEOUT', '360'))

    @staticmethod
    def _parse_json(answer):
        answer = answer.strip()
        if answer.startswith('```'):
            answer = answer.strip('`').removeprefix('json').strip()
        try:
            return json.loads(answer)
        except json.JSONDecodeError:
            repaired = repair_json(answer, return_objects=True)
            if not isinstance(repaired, dict):
                raise
            return repaired

    @staticmethod
    def _normalize_polygon(raw_polygon):
        """Normalize Seed's common 2D/3D and grouped four-point formats."""
        if not isinstance(raw_polygon, list):
            return None

        # Normal form, and the occasional [x, y, 0] form: the third value is
        # not a coordinate and must not shift all following points.
        if len(raw_polygon) >= 4 and all(
            isinstance(point, list) and len(point) >= 2
            for point in raw_polygon[:4]
        ):
            return [[point[0], point[1]] for point in raw_polygon[:4]]

        # Seed also sometimes groups two points into each inner array:
        # [[x1,y1,x2,y2], [x3,y3,x4,y4]].
        flattened = []
        for group in raw_polygon:
            if not isinstance(group, list):
                return None
            flattened.extend(group)
        if len(flattened) == 8:
            return [flattened[index:index + 2] for index in range(0, 8, 2)]
        return None

    def _recognize_image(self, width, height, labels, image_url, image_info):
        allowed_labels = json.dumps(labels, ensure_ascii=False)
        prompt = (
            '你是OCR标注模型。必须从上到下完整识别图片中所有清晰可见的文字，不能遗漏下半部分。'
            '将空间相邻、书写形式相同的连续文字合并为文字块，不要把单个汉字、拼音或标点拆成'
            '大量碎片框；块内换行使用\\n。保持原始文字、大小写、数字和标点，不要翻译、概括'
            '或补充。为每个文字块给出紧贴文字的四点多边形，坐标使用原图'
            '宽高归一化到0到1000（左上角为[0,0]，右下角为[1000,1000]），顺序为'
            f'左上、右上、右下、左下。每行必须从这些标签中选择一个：{allowed_labels}。'
            '必须理解这些标签在当前项目中的名称和语义，并依据图中文字的视觉特征逐行分类；'
            '不得创造标签，也不得固定套用其他项目的标签。'
            '不要标注被图片边缘裁断的残缺文字、空白、线条、表格边框或装饰图形。'
            '严格只输出JSON对象：'
            '{"lines":[{"text":"原文","polygon":[[x1,y1],[x2,y2],'
            '[x3,y3],[x4,y4]],"label":"标签","confidence":0.95}]}。'
            '无文字时输出{"lines":[]}。'
        )
        payload = {
            'model': self.model,
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {
                        'type': 'image_url',
                        'image_url': {'url': image_url, 'detail': 'high'},
                    },
                ],
            }],
            'temperature': 0,
            'thinking': {'type': 'disabled'},
            'response_format': {'type': 'json_object'},
            'max_tokens': 8192,
        }
        logger.info(
            'Doubao OCR request: model=%s image=%s labels=%s prompt=%s',
            self.model, image_info, labels, prompt,
        )
        response = requests.post(
            f'{self.base_url}/chat/completions',
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            },
            json=payload,
            timeout=self.timeout,
        )
        raise_for_status_with_detail(response)
        answer = response.json()['choices'][0]['message']['content']
        logger.info('Doubao OCR raw response: %s', answer)
        parsed = self._parse_json(answer)
        raw_lines = parsed.get('lines', [])
        if not isinstance(raw_lines, list):
            raise RuntimeError('Doubao OCR response field lines must be an array')

        lines = []
        for raw_line in raw_lines:
            if not isinstance(raw_line, dict):
                continue
            text = raw_line.get('text')
            label = raw_line.get('label')
            normalized_polygon = raw_line.get('polygon')
            if not text or not isinstance(normalized_polygon, list):
                logger.warning('Doubao OCR skipped line without text/polygon: %s', raw_line)
                continue
            if label not in labels:
                label = labels[0]

            normalized_points = self._normalize_polygon(normalized_polygon)
            if normalized_points is None:
                logger.warning('Doubao OCR skipped malformed polygon: %s', raw_line)
                continue
            polygon = []
            for point in normalized_points:
                try:
                    normalized_x = float(point[0])
                    normalized_y = float(point[1])
                except (TypeError, ValueError):
                    polygon = []
                    break
                x = min(1000.0, max(0.0, normalized_x)) / 1000 * width
                y = min(1000.0, max(0.0, normalized_y)) / 1000 * height
                polygon.append([x, y])
            if len(polygon) != 4:
                logger.warning('Doubao OCR skipped non-numeric polygon: %s', raw_line)
                continue
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            rectangle = {
                'x': min(xs),
                'y': min(ys),
                'width': max(xs) - min(xs),
                'height': max(ys) - min(ys),
            }
            if rectangle['width'] <= 0 or rectangle['height'] <= 0:
                continue
            lines.append({
                'text': str(text),
                'label': label,
                'polygon': polygon,
                'rectangle': rectangle,
                'probability': min(1.0, max(0.0, float(raw_line.get('confidence', 1.0)))),
            })
        logger.info(
            'Doubao OCR parsed result: count=%d lines=%s',
            len(lines), json.dumps(lines, ensure_ascii=False),
        )
        return lines

    def recognize_file(self, image_path, labels=None):
        if not self.api_key:
            raise ValueError('ARK_API_KEY is required for Doubao Seed OCR')

        labels = list(labels or ['Text'])
        mime_type = mimetypes.guess_type(image_path)[0] or 'application/octet-stream'
        with open(image_path, 'rb') as image_file:
            encoded = base64.b64encode(image_file.read()).decode('ascii')
        original_data_url = f'data:{mime_type};base64,{encoded}'
        with Image.open(image_path) as source:
            width, height = source.size
            source_format = source.format
        image_info = {
            'path': image_path,
            'mime_type': mime_type,
            'format': source_format,
            'width': width,
            'height': height,
            'bytes': len(encoded) * 3 // 4,
        }
        return self._recognize_image(
            width, height, labels, original_data_url, image_info
        )
