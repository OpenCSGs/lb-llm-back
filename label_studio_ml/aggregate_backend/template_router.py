from dataclasses import dataclass
from xml.etree import ElementTree


@dataclass(frozen=True)
class TemplateRoute:
    control_type: str
    from_name: str
    to_name: str
    data_key: str
    labels: list[str]
    transcription_from_name: str | None = None
    labels_from_name: str | None = None
    value_list: bool = False

    @property
    def is_ocr(self):
        return self.transcription_from_name is not None

    @property
    def is_multi_page(self):
        return self.control_type == 'RectangleLabels' and self.value_list


@dataclass(frozen=True)
class VQARoute:
    image_data_key: str
    questions: list[tuple[str, str, str]]
    aspect_from_name: str | None = None
    aspect_to_name: str | None = None
    labels: list[str] | None = None


def resolve_vqa_route(model):
    """Resolve Image + Text questions + TextArea answers from label config."""
    try:
        root = ElementTree.fromstring(model.label_config)
    except (AttributeError, ElementTree.ParseError, TypeError):
        return None

    image = root.find('.//Image')
    if image is None or not image.get('value', '').startswith('$'):
        return None

    texts = {
        tag.get('name'): tag.get('value', '')[1:]
        for tag in root.findall('.//Text')
        if tag.get('name') and tag.get('value', '').startswith('$')
    }
    questions = []
    for answer in root.findall('.//TextArea'):
        to_name = answer.get('toName')
        if answer.get('name') and to_name in texts:
            questions.append((texts[to_name], answer.get('name'), to_name))
    if not questions:
        return None

    aspect = root.find('.//Labels')
    aspect_from_name = aspect.get('name') if aspect is not None else None
    aspect_to_name = aspect.get('toName') if aspect is not None else None
    labels = (
        [label.get('value') for label in aspect.findall('./Label') if label.get('value')]
        if aspect is not None else []
    )
    return VQARoute(
        image_data_key=image.get('value')[1:],
        questions=questions,
        aspect_from_name=aspect_from_name,
        aspect_to_name=aspect_to_name,
        labels=labels,
    )


def _resolve_ocr_route(model):
    try:
        transcription_from_name, transcription_to_name, transcription_data_key = (
            model.get_first_tag_occurence('TextArea', 'Image')
        )
    except Exception:
        return None

    for control_type in ('RectangleLabels', 'PolygonLabels'):
        try:
            from_name, to_name, data_key = model.get_first_tag_occurence(
                control_type, 'Image'
            )
        except Exception:
            continue
        if to_name != transcription_to_name or data_key != transcription_data_key:
            continue
        labels = list(model.label_interface.get_tag(from_name).labels)
        if not labels:
            raise ValueError(f'{control_type} must contain at least one Label')
        return TemplateRoute(
            control_type=control_type,
            from_name=from_name,
            to_name=to_name,
            data_key=data_key,
            labels=labels,
            transcription_from_name=transcription_from_name,
        )

    try:
        labels_from_name, labels_to_name, labels_data_key = (
            model.get_first_tag_occurence('Labels', 'Image')
        )
    except Exception:
        labels_from_name = None
    if labels_from_name:
        labels = list(model.label_interface.get_tag(labels_from_name).labels)
        if not labels:
            raise ValueError('OCR Labels must contain at least one Label')
        for control_type in ('Rectangle', 'Polygon'):
            try:
                from_name, to_name, data_key = model.get_first_tag_occurence(
                    control_type, 'Image'
                )
            except Exception:
                continue
            if not (
                to_name == transcription_to_name == labels_to_name
                and data_key == transcription_data_key == labels_data_key
            ):
                continue
            return TemplateRoute(
                control_type=control_type,
                from_name=from_name,
                to_name=to_name,
                data_key=data_key,
                labels=labels,
                transcription_from_name=transcription_from_name,
                labels_from_name=labels_from_name,
            )

    raise ValueError(
        'OCR config requires a supported region control and a per-region '
        'TextArea targeting the same Image'
    )


def _resolve_image_data_source(model, to_name, fallback_data_key):
    """Return the XML data key and whether the target Image uses valueList."""
    try:
        root = ElementTree.fromstring(model.label_config)
    except (AttributeError, ElementTree.ParseError, TypeError):
        return fallback_data_key, False

    for image in root.findall('.//Image'):
        if image.get('name') != to_name:
            continue
        value_list = image.get('valueList', '')
        if value_list.startswith('$'):
            return value_list[1:], True
        value = image.get('value', '')
        if value.startswith('$'):
            return value[1:], False
    return fallback_data_key, False


def resolve_image_route(model) -> TemplateRoute:
    """Resolve an image annotation pipeline from project config."""
    ocr_route = _resolve_ocr_route(model)
    if ocr_route:
        return ocr_route

    for control_type in ('BrushLabels', 'PolygonLabels', 'RectangleLabels'):
        try:
            from_name, to_name, data_key = model.get_first_tag_occurence(
                control_type, 'Image'
            )
        except Exception:
            continue
        labels = list(model.label_interface.get_tag(from_name).labels)
        if not labels:
            raise ValueError(f'{control_type} must contain at least one Label')
        data_key, value_list = _resolve_image_data_source(
            model, to_name, data_key
        )
        return TemplateRoute(
            control_type=control_type,
            from_name=from_name,
            to_name=to_name,
            data_key=data_key,
            labels=labels,
            value_list=value_list,
        )

    raise ValueError(
        'Unsupported labeling config: expected BrushLabels, PolygonLabels, '
        'RectangleLabels, multi-page document, OCR, or '
        'visual-question-answering controls'
    )
