import numpy as np

from label_studio_ml.aggregate_backend.result_converter import convert_masks
from label_studio_ml.aggregate_backend.template_router import resolve_image_route


class _Tag:
    labels = ['文本', '手写体']


class _LabelInterface:
    def get_tag(self, _name):
        return _Tag()


class _Model:
    label_interface = _LabelInterface()

    def __init__(self, controls):
        self.controls = controls

    def get_first_tag_occurence(self, control_type, _object_type):
        if control_type not in self.controls:
            raise ValueError(control_type)
        return self.controls[control_type]


def test_routes_brush_semantic_segmentation():
    route = resolve_image_route(
        _Model({'BrushLabels': ('brush', 'image', 'image')})
    )
    assert route.control_type == 'BrushLabels'
    assert route.labels == ['文本', '手写体']


def test_routes_ocr_before_brush():
    route = resolve_image_route(_Model({
        'BrushLabels': ('brush', 'image', 'image'),
        'RectangleLabels': ('bbox', 'image', 'image'),
        'TextArea': ('transcription', 'image', 'image'),
    }))
    assert route.is_ocr
    assert route.from_name == 'bbox'


def test_brush_result_is_rle():
    route = resolve_image_route(
        _Model({'BrushLabels': ('brush', 'image', 'image')})
    )
    results, score = convert_masks(
        route,
        [np.array([[0, 1], [1, 0]], dtype=np.uint8)],
        [0.9],
        ['文本'],
        2,
        2,
    )
    assert results[0]['type'] == 'brushlabels'
    assert results[0]['value']['brushlabels'] == ['文本']
    assert score == 0.9
