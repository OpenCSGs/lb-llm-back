from dataclasses import dataclass


@dataclass(frozen=True)
class TemplateRoute:
    control_type: str
    from_name: str
    to_name: str
    data_key: str
    labels: list[str]
    transcription_from_name: str | None = None
    labels_from_name: str | None = None

    @property
    def is_ocr(self):
        return self.transcription_from_name is not None


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


def resolve_image_route(model) -> TemplateRoute:
    """Resolve the enabled BrushLabels or OCR pipeline from project config."""
    ocr_route = _resolve_ocr_route(model)
    if ocr_route:
        return ocr_route

    try:
        from_name, to_name, data_key = model.get_first_tag_occurence(
            'BrushLabels', 'Image'
        )
    except Exception as exc:
        raise ValueError(
            'Unsupported labeling config: this release supports BrushLabels '
            'semantic segmentation and OCR templates only'
        ) from exc

    labels = list(model.label_interface.get_tag(from_name).labels)
    if not labels:
        raise ValueError('BrushLabels must contain at least one Label')
    return TemplateRoute(
        control_type='BrushLabels',
        from_name=from_name,
        to_name=to_name,
        data_key=data_key,
        labels=labels,
    )
