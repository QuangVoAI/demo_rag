from __future__ import annotations

import uuid


class RequestContextMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request_id = (
            request.headers.get("X-Request-ID")
            or request.headers.get("X-Correlation-ID")
            or str(uuid.uuid4())
        )
        request.request_id = request_id
        request.correlation_id = request.headers.get("X-Correlation-ID") or request_id

        response = self.get_response(request)
        response["X-Request-ID"] = request_id
        response["X-Correlation-ID"] = request.correlation_id
        return response
