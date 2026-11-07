# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sonar Exception handlers are registered in four layers:
1. framework errors raised by FastAPI/Starlette
2. Sonar-specific errors dispatched via a single ``SonarError`` handler
3. fallback handlers for raw exceptions not yet migrated to ``SonarError``
4. the raw ``Exception`` handler as a safety net
Registering specific exception types (rather than only ``Exception``)
ensures they are handled by ``ExceptionMiddleware`` (inside the Prometheus
middleware) rather than ``ServerErrorMiddleware`` (outside it), so their
status codes are recorded correctly.
"""

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError

from sonar.exceptions import SonarError

from .handlers.exception import exception_handler
from .handlers.http import http_exception_handler
from .handlers.sonar_error import sonar_error_handler
from .handlers.validation import validation_exception_handler


def init_exception_handler(app: FastAPI):
    #   1. framework errors raised by FastAPI/Starlette
    app.exception_handler(HTTPException)(http_exception_handler)
    app.exception_handler(RequestValidationError)(validation_exception_handler)

    #   2. Sonar-specific errors dispatched via a single ``SonarError`` handler
    app.exception_handler(SonarError)(sonar_error_handler)

    #   3. fallback handlers for raw exceptions not yet migrated to ``SonarError``
    # TODO(zqzten): remove these fallback handlers after migration to SonarError
    app.exception_handler(ValueError)(exception_handler)
    app.exception_handler(TypeError)(exception_handler)
    app.exception_handler(OverflowError)(exception_handler)
    app.exception_handler(NotImplementedError)(exception_handler)

    #   4. the raw ``Exception`` handler as a safety net
    app.exception_handler(Exception)(exception_handler)
