"""
app/core/exceptions.py
-----------------------
Custom exception classes + FastAPI exception handlers.
Register the handlers in main.py with app.add_exception_handler(…).
"""

from fastapi import Request
from fastapi.responses import JSONResponse


# ── Domain exceptions ──────────────────────────────────────────────────────

class HimakBaseException(Exception):
    """Root exception for all Himak-specific errors."""
    status_code: int = 500
    detail: str = "An unexpected error occurred."

    def __init__(self, detail: str | None = None):
        self.detail = detail or self.__class__.detail
        super().__init__(self.detail)


class EmbeddingError(HimakBaseException):
    status_code = 503
    detail = "Failed to generate query embedding."


class VectorStoreError(HimakBaseException):
    status_code = 503
    detail = "Vector database query failed."


class ValidationError(HimakBaseException):
    status_code = 422
    detail = "Invalid request payload."


class RateLimitError(HimakBaseException):
    status_code = 429
    detail = "Too many requests. Please slow down."


# ── FastAPI exception handlers ─────────────────────────────────────────────

async def himak_exception_handler(request: Request, exc: HimakBaseException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.__class__.__name__,
                "message": exc.detail,
            },
        },
    )


async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "InternalServerError",
                "message": "An unexpected server error occurred.",
            },
        },
    )