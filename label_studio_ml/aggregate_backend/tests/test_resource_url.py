from label_studio_ml.aggregate_backend.resource_url import normalize_ls_resource_url


def test_strips_matching_label_studio_subpath():
    assert normalize_ls_resource_url(
        '/-/label-studio/data/upload/81/image.png',
        'http://modelhub.cmr-co.com/-/label-studio',
    ) == '/data/upload/81/image.png'


def test_preserves_query_when_stripping_subpath():
    assert normalize_ls_resource_url(
        '/-/label-studio/data/local-files?d=images/example.png',
        'http://modelhub.cmr-co.com/-/label-studio/',
    ) == '/data/local-files?d=images/example.png'


def test_no_subpath_environment_is_unchanged():
    assert normalize_ls_resource_url(
        '/data/upload/81/image.png',
        'http://192.168.3.17:8080',
    ) == '/data/upload/81/image.png'


def test_complete_external_url_is_unchanged():
    url = 'https://cdn.example.com/-/label-studio/data/upload/image.png'
    assert normalize_ls_resource_url(
        url, 'http://modelhub.cmr-co.com/-/label-studio'
    ) == url


def test_non_matching_or_unknown_path_is_unchanged():
    assert normalize_ls_resource_url(
        '/another-prefix/data/upload/81/image.png',
        'http://modelhub.cmr-co.com/-/label-studio',
    ) == '/another-prefix/data/upload/81/image.png'
    assert normalize_ls_resource_url(
        '/-/label-studio/api/projects/1',
        'http://modelhub.cmr-co.com/-/label-studio',
    ) == '/-/label-studio/api/projects/1'


def test_repairs_malformed_hostless_http_url():
    assert normalize_ls_resource_url(
        'http:///-/label-studio/data/upload/81/image.png',
        'http://modelhub.cmr-co.com/-/label-studio',
    ) == '/data/upload/81/image.png'
