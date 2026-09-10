"""Safe, user-facing error details for third-party inference APIs."""


class ProviderAPIError(RuntimeError):
    pass


def raise_for_status_with_detail(response):
    if response.ok:
        return
    message = None
    try:
        body = response.json()
        if isinstance(body, dict):
            error = body.get('error')
            message = body.get('message') or body.get('detail')
            if not message and isinstance(error, dict):
                message = error.get('message') or error.get('code')
            elif not message and isinstance(error, str):
                message = error
    except ValueError:
        pass
    message = str(message or 'third-party model request failed')[:1000]
    raise ProviderAPIError(f'Third-party model HTTP {response.status_code}: {message}')
