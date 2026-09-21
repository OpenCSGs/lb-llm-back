from uuid import uuid4

from label_studio_sdk.converter import brush

from .templates.object_detection import mask_to_rectangle
from .templates.polygon_segmentation import mask_to_polygons

def _common_result(route, width, height, probability):
    return {
        'id': str(uuid4())[:8],
        'from_name': route.from_name,
        'to_name': route.to_name,
        'original_width': width,
        'original_height': height,
        'image_rotation': 0,
        'score': probability,
        'readonly': False,
    }


def convert_masks(route, masks, probabilities, labels, width, height):
    """Convert model-neutral masks into results for the selected LS control."""
    results = []
    used_probabilities = []
    for mask, probability, label in zip(masks, probabilities, labels):
        common = _common_result(route, width, height, probability)
        if route.control_type == 'BrushLabels':
            results.append({
                **common,
                'type': 'brushlabels',
                'value': {
                    'format': 'rle',
                    'rle': brush.mask2rle(mask * 255),
                    'brushlabels': [label],
                },
            })
            used_probabilities.append(probability)
        elif route.control_type == 'RectangleLabels':
            rectangle = mask_to_rectangle(mask)
            if rectangle:
                results.append({
                    **common,
                    'type': 'rectanglelabels',
                    'value': {
                        **rectangle,
                        'rotation': 0,
                        'rectanglelabels': [label],
                    },
                })
                used_probabilities.append(probability)
        elif route.control_type == 'PolygonLabels':
            for points in mask_to_polygons(mask):
                results.append({
                    **common,
                    'id': str(uuid4())[:8],
                    'type': 'polygonlabels',
                    'value': {
                        'points': points,
                        'closed': True,
                        'polygonlabels': [label],
                    },
                })
                used_probabilities.append(probability)

    score = sum(used_probabilities) / max(len(used_probabilities), 1)
    return results, score


def convert_ocr_lines(route, lines, width, height):
    """Convert OCR text lines to linked regions and per-region TextArea results."""
    results = []
    scores = []
    for line in lines:
        label = line.get('label') if line.get('label') in route.labels else route.labels[0]
        score = line['probability']
        region_id = str(uuid4())[:8]
        common = {
            'id': region_id,
            'to_name': route.to_name,
            'original_width': width,
            'original_height': height,
            'image_rotation': 0,
            'score': score,
            'readonly': False,
        }

        if route.control_type in ('PolygonLabels', 'Polygon') and line.get('polygon'):
            points = [
                [round(float(point[0]) / width * 100, 4),
                 round(float(point[1]) / height * 100, 4)]
                for point in line['polygon']
            ]
            region_value = {'points': points}
            if route.control_type == 'PolygonLabels':
                region_value.update({'polygonlabels': [label], 'closed': True})
                region_type = 'polygonlabels'
            else:
                region_type = 'polygon'
        else:
            rectangle = line.get('rectangle')
            if not rectangle:
                continue
            region_value = {
                'x': round(float(rectangle['x']) / width * 100, 4),
                'y': round(float(rectangle['y']) / height * 100, 4),
                'width': round(float(rectangle['width']) / width * 100, 4),
                'height': round(float(rectangle['height']) / height * 100, 4),
                'rotation': 0,
            }
            if route.control_type == 'RectangleLabels':
                region_value['rectanglelabels'] = [label]
                region_type = 'rectanglelabels'
            else:
                region_type = 'rectangle'

        textarea_value = {
            **{key: value for key, value in region_value.items()
               if key not in ('rectanglelabels', 'polygonlabels', 'closed')},
            'text': [line['text']],
        }
        results.append({
            **common,
            'from_name': route.from_name,
            'type': region_type,
            'value': region_value,
        })
        if route.labels_from_name:
            results.append({
                **common,
                'from_name': route.labels_from_name,
                'type': 'labels',
                'value': {
                    **{key: value for key, value in region_value.items()
                       if key not in ('rectanglelabels', 'polygonlabels', 'closed')},
                    'labels': [label],
                },
            })
        results.append({
            **common,
            'from_name': route.transcription_from_name,
            'type': 'textarea',
            'value': textarea_value,
        })
        scores.append(score)

    return results, sum(scores) / max(len(scores), 1)


def convert_vqa(route, answers, aspect, score):
    """Convert one Seed VQA response into Label Studio textarea/label results."""
    results = []
    for question_key, answer_from_name, question_to_name in route.questions:
        if question_key not in answers:
            continue
        results.append({
            'id': str(uuid4())[:8],
            'from_name': answer_from_name,
            'to_name': question_to_name,
            'type': 'textarea',
            'score': score,
            'readonly': False,
            'value': {'text': [answers[question_key]]},
        })
    if aspect and route.aspect_from_name and route.aspect_to_name:
        results.append({
            'id': str(uuid4())[:8],
            'from_name': route.aspect_from_name,
            'to_name': route.aspect_to_name,
            'type': 'labels',
            'score': score,
            'readonly': False,
            'value': {'labels': [aspect]},
        })
    return results


def convert_document_regions(route, regions):
    """Convert multi-page regions, preserving LS's zero-based item index."""
    results = []
    scores = []
    for region in regions:
        probability = float(region.get('confidence', 1.0))
        common = _common_result(
            route,
            int(region['original_width']),
            int(region['original_height']),
            probability,
        )
        results.append({
            **common,
            'type': 'rectanglelabels',
            'item_index': int(region['page_index']),
            'value': {
                'x': float(region['x']),
                'y': float(region['y']),
                'width': float(region['width']),
                'height': float(region['height']),
                'rotation': 0,
                'rectanglelabels': [region['label']],
            },
        })
        scores.append(probability)
    return results, sum(scores) / max(len(scores), 1)
