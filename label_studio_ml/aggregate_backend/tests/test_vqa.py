import json

from label_studio_ml.aggregate_backend.templates.vqa import DoubaoVQA


class FakeResponse:
    ok = True

    def raise_for_status(self):
        return None

    def json(self):
        return {
            'choices': [{
                'message': {
                    'content': json.dumps({
                        'answers': [
                            {'key': 'question1', 'answer': '两只猫'},
                            {'key': 'unknown', 'answer': 'ignored'},
                        ],
                        'aspect': 'counting',
                    })
                }
            }]
        }


def test_vqa_uses_one_request_and_filters_result(monkeypatch, tmp_path):
    image_path = tmp_path / 'image.png'
    image_path.write_bytes(b'fake-image')
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs['json'])
        return FakeResponse()

    monkeypatch.setattr(
        'label_studio_ml.aggregate_backend.templates.vqa.requests.post', fake_post
    )
    answers, aspect, score = DoubaoVQA(api_key='test-key').answer_file(
        str(image_path),
        {'question1': '图片里有几只猫？'},
        ['counting', 'comparison'],
    )

    assert len(calls) == 1
    assert answers == {'question1': '两只猫'}
    assert aspect == 'counting'
    assert score == 1.0
