import base64
import json
import logging
import mimetypes
import os

from json_repair import repair_json
import requests
from ..providers.errors import raise_for_status_with_detail


logger = logging.getLogger(__name__)


class DoubaoVQA:
    """Answer all questions about one image in a single Seed request."""

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
            parsed = repair_json(answer, return_objects=True)
            if not isinstance(parsed, dict):
                raise
            return parsed

    def answer_file(self, image_path, questions, aspect_labels=None):
        if not self.api_key:
            raise ValueError('ARK_API_KEY is required for Doubao Seed VQA')
        aspect_labels = list(aspect_labels or [])
        question_payload = [
            {'key': key, 'question': str(question)}
            for key, question in questions.items()
            if question is not None and str(question).strip()
        ]
        if not question_payload:
            return {}, None, 1.0

        mime_type = mimetypes.guess_type(image_path)[0] or 'application/octet-stream'
        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()
        image_url = 'data:{};base64,{}'.format(
            mime_type, base64.b64encode(image_bytes).decode('ascii')
        )
        aspect_instruction = ''
        aspect_schema = ''
        if aspect_labels:
            aspect_instruction = (
                '同时判断问题所属类型，aspect必须且只能从以下项目标签中选择一个：'
                f'{json.dumps(aspect_labels, ensure_ascii=False)}。'
            )
            aspect_schema = ',"aspect":"项目标签"'
        prompt = (
            '你是视觉问答模型。请根据图片内容逐一回答下列问题。答案必须准确、简洁，'
            '不要编造图片中无法判断的信息；无法判断时明确回答“无法判断”。'
            f'{aspect_instruction}问题列表：'
            f'{json.dumps(question_payload, ensure_ascii=False)}。'
            '严格只输出JSON对象，保持每个key原样返回：'
            '{"answers":[{"key":"问题key","answer":"答案"}]'
            f'{aspect_schema}' + '}。'
        )
        logger.info(
            'Doubao VQA request: model=%s image=%s bytes=%d questions=%s labels=%s',
            self.model, image_path, len(image_bytes), question_payload, aspect_labels,
        )
        response = requests.post(
            f'{self.base_url}/chat/completions',
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            },
            json={
                'model': self.model,
                'messages': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': image_url, 'detail': 'high'}},
                    ],
                }],
                'temperature': 0,
                'thinking': {'type': 'disabled'},
                'response_format': {'type': 'json_object'},
                'max_tokens': 4096,
            },
            timeout=self.timeout,
        )
        raise_for_status_with_detail(response)
        raw_answer = response.json()['choices'][0]['message']['content']
        logger.info('Doubao VQA raw response: %s', raw_answer)
        parsed = self._parse_json(raw_answer)
        answers = {}
        for item in parsed.get('answers', []):
            if isinstance(item, dict) and item.get('key') in questions:
                answers[item['key']] = str(item.get('answer', ''))
        aspect = parsed.get('aspect')
        if aspect not in aspect_labels:
            aspect = None
        logger.info('Doubao VQA parsed result: answers=%s aspect=%s', answers, aspect)
        return answers, aspect, 1.0
