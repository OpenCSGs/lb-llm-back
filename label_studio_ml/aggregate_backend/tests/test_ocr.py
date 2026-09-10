from label_studio_ml.aggregate_backend.templates.ocr import DoubaoOCR


def test_normalize_seed_grouped_polygon():
    polygon = DoubaoOCR._normalize_polygon([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert polygon == [[1, 2], [3, 4], [5, 6], [7, 8]]
