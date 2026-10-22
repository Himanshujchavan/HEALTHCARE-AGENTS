"""Normalized error handling for the API.

Every rejected request answers with the same envelope::

    {
        "status": "error",
        "error": "validation_error" | "http_error" | "internal_error",
        "message": "...",
        "details": [{"field": "...", "message": "...", "code": "..."}]
    }

The ``detail`` key is kept alongside ``message`` so existing clients that
read FastAPI's default shape keep working.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return a normalized error envelope for rejected input.

    Every problem is reported per field, so a client can point at the exact
    parameter that was missing, invalid or out of range.
    """
    details = []
    for error in exc.errors():
        location = [
            str(part) for part in error.get("loc", [])
            if part not in ("body", "query", "path", "header", "cookie")
        ]
        details.append({
            "field": ".".join(location) or None,
            "message": error.get("msg", "Invalid value"),
            "code": error.get("type"),
        })

    logger.info("Validation error on %s: %s", request.url.path, details)
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "error": "validation_error",
            "message": "The submitted data is invalid.",
            "details": details,
        },
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Keep the error shape consistent for every HTTP error."""
    return JSONResponse(
        status_code=exc.status_code,
        headers=getattr(exc, "headers", None),
        content={
            "status": "error",
            "error": "http_error",
            "message": exc.detail,
            "detail": exc.detail,
        },
    )


async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler for unexpected failures."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": "internal_error",
            "message": "Internal server error",
            "detail": str(exc),
        },
    )


def install_error_handlers(app: FastAPI) -> FastAPI:
    """Register the normalized error handlers on an application."""
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)
    return app
