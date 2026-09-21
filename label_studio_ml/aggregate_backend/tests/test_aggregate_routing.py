import numpy as np

from label_studio_ml.aggregate_backend.result_converter import (
    convert_document_regions,
    convert_masks,
    convert_vqa,
)
from label_studio_ml.aggregate_backend.template_router import (
    resolve_image_route,
    resolve_vqa_route,
)


class _Tag:
    labels = ['文本', '手写体']


class _LabelInterface:
    def get_tag(self, _name):
        return _Tag()


class _Model:
    label_interface = _LabelInterface()

    def __init__(self, controls):
        self.controls = controls
        self.label_config = None

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


def test_routes_object_detection():
    route = resolve_image_route(
        _Model({'RectangleLabels': ('objects', 'image', 'image')})
    )
    assert route.control_type == 'RectangleLabels'


def test_routes_polygon_segmentation():
    route = resolve_image_route(
        _Model({'PolygonLabels': ('polygons', 'image', 'image')})
    )
    assert route.control_type == 'PolygonLabels'


def test_routes_multi_page_document_from_image_value_list():
    model = _Model({'RectangleLabels': ('rectangles', 'pdf', 'pages')})
    model.label_config = '''
    <View>
      <RectangleLabels name="rectangles" toName="pdf">
        <Label value="Title"/><Label value="Date"/>
      </RectangleLabels>
      <Image valueList="$pages" name="pdf"/>
    </View>
    '''
    route = resolve_image_route(model)
    assert route.is_multi_page
    assert route.data_key == 'pages'


def test_multi_page_results_include_top_level_item_index():
    model = _Model({'RectangleLabels': ('rectangles', 'pdf', 'pages')})
    model.label_config = '''
    <View>
      <RectangleLabels name="rectangles" toName="pdf">
        <Label value="Title"/>
      </RectangleLabels>
      <Image valueList="$pages" name="pdf"/>
    </View>
    '''
    route = resolve_image_route(model)
    results, score = convert_document_regions(route, [{
        'page_index': 3,
        'label': 'Title',
        'x': 10,
        'y': 20,
        'width': 30,
        'height': 40,
        'confidence': 0.9,
        'original_width': 1200,
        'original_height': 1600,
    }])
    assert results[0]['item_index'] == 3
    assert results[0]['type'] == 'rectanglelabels'
    assert results[0]['value']['rectanglelabels'] == ['Title']
    assert results[0]['original_width'] == 1200
    assert score == 0.9


def test_mask_results_convert_to_rectangle_and_polygon():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 4:12] = 1
    rectangle_route = resolve_image_route(
        _Model({'RectangleLabels': ('objects', 'image', 'image')})
    )
    rectangle_results, rectangle_score = convert_masks(
        rectangle_route, [mask], [0.8], ['文本'], 20, 20
    )
    assert rectangle_results[0]['type'] == 'rectanglelabels'
    assert rectangle_results[0]['value']['rectanglelabels'] == ['文本']
    assert rectangle_score == 0.8

    polygon_route = resolve_image_route(
        _Model({'PolygonLabels': ('polygons', 'image', 'image')})
    )
    polygon_results, polygon_score = convert_masks(
        polygon_route, [mask], [0.7], ['文本'], 20, 20
    )
    assert polygon_results[0]['type'] == 'polygonlabels'
    assert polygon_results[0]['value']['polygonlabels'] == ['文本']
    assert polygon_score == 0.7


def test_routes_and_converts_vqa():
    model = _Model({})
    model.label_config = '''
    <View>
      <Image name="image" value="$image"/>
      <Labels name="aspect" toName="q1"><Label value="counting"/></Labels>
      <Text name="q1" value="$question1"/>
      <TextArea name="answer1" toName="q1"/>
      <Text name="q2" value="$question2"/>
      <TextArea name="answer2" toName="q2"/>
    </View>
    '''
    route = resolve_vqa_route(model)
    assert route.image_data_key == 'image'
    assert route.questions == [
        ('question1', 'answer1', 'q1'),
        ('question2', 'answer2', 'q2'),
    ]
    results = convert_vqa(
        route,
        {'question1': '两个', 'question2': '蓝色'},
        'counting',
        1.0,
    )
    assert [result['type'] for result in results] == [
        'textarea', 'textarea', 'labels'
    ]
    assert results[0]['value']['text'] == ['两个']
    assert results[-1]['value']['labels'] == ['counting']
