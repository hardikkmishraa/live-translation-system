from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from backend.app.schemas.errors import APIErrorResponse, APIErrorDetail
from backend.app.utils.logger import logger

async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Formats HTTPException into consistent APIErrorResponse schema."""
    error_payload = APIErrorResponse(
        success=False,
        error=APIErrorDetail(
            code=f"HTTP_{exc.status_code}",
            message=str(exc.detail),
            diagnostics={"path": request.url.path}
        )
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload.model_dump()
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Formats Pydantic validation errors cleanly without leaking internals."""
    errors = exc.errors()
    first_error = errors[0] if errors else {}
    msg = first_error.get("msg", "Invalid request parameter")
    field = " -> ".join(str(loc) for loc in first_error.get("loc", []))

    error_payload = APIErrorResponse(
        success=False,
        error=APIErrorDetail(
            code="VALIDATION_ERROR",
            message=f"{field}: {msg}",
            field=field,
            diagnostics={"total_errors": len(errors)}
        )
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_payload.model_dump()
    )

async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catches unhandled errors and prevents internal stack traces from leaking."""
    logger.error(f"Unhandled server error on {request.url.path}: {exc}", exc_info=True)
    error_payload = APIErrorResponse(
        success=False,
        error=APIErrorDetail(
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected server error occurred during processing. Please try again.",
            diagnostics={"path": request.url.path}
        )
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_payload.model_dump()
    )
