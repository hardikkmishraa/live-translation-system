import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from backend.app.utils.logger import logger

class PrivacyPreservingLoggingMiddleware(BaseHTTPMiddleware):
    """
    HTTP Middleware that logs request metadata (method, path, status, latency)
    while strictly omitting user speech transcripts and binary audio buffers.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        t_start = time.perf_counter()
        method = request.method
        path = request.url.path
        client_host = request.client.host if request.client else "unknown"

        try:
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0

            # Exclude noisy polling from verbose logs
            if path != "/api/v1/health" and path != "/api/health":
                logger.info(
                    f"{method} {path} - {response.status_code} ({elapsed_ms:.1f}ms) [Client: {client_host}]"
                )
            return response
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            logger.error(
                f"{method} {path} - ERROR ({elapsed_ms:.1f}ms) [Client: {client_host}]: {exc.__class__.__name__}"
            )
            raise
