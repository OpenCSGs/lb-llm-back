import json

from PIL import Image

from label_studio_ml.aggregate_backend.templates.multi_page_document import (
    DoubaoDocument,
)


class FakeResponse:
    ok = True

    def __init__(self, page_count):
        regions = [
            {
                'page_index': index,
                'label': 'Title',
                'bbox': [100, 200, 300, 400],
                'confidence': 0.9,
            }
            for index in range(page_count)
        ]
        self._body = {
            'choices': [{'message': {'content': json.dumps({'regions': regions})}}]
        }
        self.status_code = 200
        self.text = json.dumps(self._body)

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


def test_document_pages_are_batched_and_page_indexes_are_merged(
    monkeypatch, tmp_path
):
    monkeypatch.setenv('ARK_API_KEY', 'test-key')
    monkeypatch.setenv('DOUBAO_DOCUMENT_BATCH_SIZE', '10')
    page_paths = []
    for index in range(25):
        path = tmp_path / f'page-{index}.png'
        Image.new('RGB', (100 + index, 200 + index)).save(path)
        page_paths.append(str(path))

    requests = []

    def fake_post(url, headers, json, timeout):
        image_count = sum(
            item['type'] == 'image_url'
            for item in json['messages'][0]['content']
        )
        requests.append(image_count)
        return FakeResponse(image_count)

    monkeypatch.setattr(
        'label_studio_ml.aggregate_backend.templates.multi_page_document.requests.post',
        fake_post,
    )

    regions = DoubaoDocument().extract(page_paths, ['Title'])

    assert requests == [10, 10, 5]
    assert [region['page_index'] for region in regions] == list(range(25))
    assert regions[10]['original_width'] == 110
    assert regions[24]['original_height'] == 224


def test_document_filters_invalid_model_regions(monkeypatch, tmp_path):
    monkeypatch.setenv('ARK_API_KEY', 'test-key')
    page = tmp_path / 'page.png'
    Image.new('RGB', (100, 200)).save(page)

    class InvalidRegionResponse(FakeResponse):
        def __init__(self):
            body = {'regions': [
                {'page_index': 0, 'label': 'Unknown', 'bbox': [0, 0, 10, 10]},
                {'page_index': 1, 'label': 'Title', 'bbox': [0, 0, 10, 10]},
                {'page_index': 0, 'label': 'Title', 'bbox': [20, 20, 10, 10]},
                {'page_index': 0, 'label': 'Title', 'bbox': [-20, 100, 1200, 900],
                 'confidence': 2},
            ]}
            self._body = {
                'choices': [{'message': {'content': json.dumps(body)}}]
            }
            self.status_code = 200
            self.text = json.dumps(self._body)

    monkeypatch.setattr(
        'label_studio_ml.aggregate_backend.templates.multi_page_document.requests.post',
        lambda *args, **kwargs: InvalidRegionResponse(),
    )
    regions = DoubaoDocument().extract([str(page)], ['Title'])
    assert len(regions) == 1
    assert regions[0]['page_index'] == 0
    assert regions[0]['x'] == 0
    assert regions[0]['width'] == 100
    assert regions[0]['confidence'] == 1


def test_document_requires_api_key(monkeypatch):
    monkeypatch.delenv('ARK_API_KEY', raising=False)
    document = DoubaoDocument(api_key='')
    document.api_key = None
    try:
        document.extract(['page.png'], ['Title'])
    except ValueError as error:
        assert 'ARK_API_KEY' in str(error)
    else:
        raise AssertionError('missing API key must fail before the request')
