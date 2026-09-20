"""Label Studio resource URL normalization helpers."""

import logging
from urllib.parse import urlsplit, urlunsplit


logger = logging.getLogger(__name__)


LS_RESOURCE_PREFIXES = (
    '/data/upload/',
    '/data/local-files',
    '/storage-data/',
)


def normalize_ls_resource_url(resource_url, ls_url):
    """Remove a duplicated LS deployment subpath before SDK URL resolution.

    Label Studio includes ``FORCE_SCRIPT_NAME`` in uploaded-file values when it
    is hosted below a subpath. The SDK only recognizes paths beginning with
    ``/data/upload`` (and related LS resource paths), so strip the subpath once
    when it exactly matches the path component of ``ls_url``.
    """
    logger.info(
        'LS resource URL step 1/input: ls_url=%r resource_url=%r',
        ls_url,
        resource_url,
    )
    if not resource_url or not ls_url:
        logger.info(
            'LS resource URL step 2/skip: missing ls_url or resource_url; output=%r',
            resource_url,
        )
        return resource_url

    ls = urlsplit(ls_url)
    resource = urlsplit(resource_url)
    ls_path = ls.path.rstrip('/')
    resource_path = resource.path
    logger.info(
        'LS resource URL step 2/parse: ls_scheme=%r ls_host=%r '
        'ls_path=%r resource_scheme=%r resource_host=%r resource_path=%r',
        ls.scheme,
        ls.netloc,
        ls_path,
        resource.scheme,
        resource.netloc,
        resource_path,
    )

    # A valid absolute URL is already independently resolvable.
    if resource.scheme in ('http', 'https') and resource.netloc:
        logger.info(
            'LS resource URL step 3/keep: resource is a complete absolute URL; '
            'output=%r',
            resource_url,
        )
        return resource_url

    stripped_subpath = False

    if ls_path:
        if resource_path == ls_path:
            resource_path = '/'
            stripped_subpath = True
            logger.info(
                'LS resource URL step 3/strip: resource equals LS subpath; '
                'stripped_path=%r',
                resource_path,
            )
        elif resource_path.startswith(ls_path + '/'):
            resource_path = resource_path[len(ls_path):]
            stripped_subpath = True
            logger.info(
                'LS resource URL step 3/strip: removed matching LS subpath %r; '
                'stripped_path=%r',
                ls_path,
                resource_path,
            )
        else:
            logger.info(
                'LS resource URL step 3/keep: resource path does not start with '
                'LS subpath %r; output=%r',
                ls_path,
                resource_url,
            )
            return resource_url
    else:
        logger.info('LS resource URL step 3/keep: LS URL has no deployment subpath')

    if not resource_path.startswith(LS_RESOURCE_PREFIXES):
        logger.info(
            'LS resource URL step 4/keep: normalized path is not a known LS '
            'resource path; path=%r output=%r',
            resource_path,
            resource_url,
        )
        return resource_url

    # Preserve a normal relative resource unchanged when no subpath was
    # removed. A malformed absolute URL such as http:///data/upload/... is
    # intentionally reduced to its path so the SDK can rebuild it from ls_url.
    if not stripped_subpath and not resource.scheme:
        logger.info(
            'LS resource URL step 4/keep: relative LS resource is already '
            'normalized; output=%r',
            resource_url,
        )
        return resource_url

    normalized = urlunsplit(
        ('', '', resource_path, resource.query, resource.fragment)
    )
    resolved_preview = ls_url.rstrip('/') + '/' + normalized.lstrip('/')
    logger.info(
        'LS resource URL step 4/output: normalized=%r resolved_preview=%r',
        normalized,
        resolved_preview,
    )
    return normalized
