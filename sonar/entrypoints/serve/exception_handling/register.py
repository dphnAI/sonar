# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Aphrodite Exception handlers are registered in four layers:
1. framework errors raised by FastAPI/Starlette
2. Aphrodite-specific errors dispatched via a single ``AphroditeError`` handler
3. fallback handlers for raw exceptions not yet migrated to ``AphroditeError``
4. the raw ``Exception`` handler as a safety net
Registering specific exception types (rather than only ``Exception``)
ensures they are handled by ``ExceptionMiddleware`` (inside the Prometheus
middleware) rather than ``ServerErrorMiddleware`` (outside it), so their
status codes are recorded correctly.
"""

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError

from aphrodite.exceptions import AphroditeError

from .handlers.aphrodite_error import aphrodite_error_handler
from .handlers.exception import exception_handler
from .handlers.http import http_exception_handler
from .handlers.validation import validation_exception_handler


def init_exception_handler(app: FastAPI):
    #   1. framework errors raised by FastAPI/Starlette
    app.exception_handler(HTTPException)(http_exception_handler)
    app.exception_handler(RequestValidationError)(validation_exception_handler)

    #   2. Aphrodite-specific errors dispatched via a single ``AphroditeError`` handler
    app.exception_handler(AphroditeError)(aphrodite_error_handler)

    #   3. fallback handlers for raw exceptions not yet migrated to ``AphroditeError``
    # TODO(zqzten): remove these fallback handlers after migration to AphroditeError
    app.exception_handler(ValueError)(exception_handler)
    app.exception_handler(TypeError)(exception_handler)
    app.exception_handler(OverflowError)(exception_handler)
    app.exception_handler(NotImplementedError)(exception_handler)

    #   4. the raw ``Exception`` handler as a safety net
    app.exception_handler(Exception)(exception_handler)
