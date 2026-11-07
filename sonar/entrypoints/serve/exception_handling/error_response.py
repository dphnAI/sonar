# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

from aphrodite.exceptions import GenerationError, RetryableRequestError
from aphrodite.logger import init_logger

from ..engine.protocol import ErrorInfo, ErrorResponse
from .utils import sanitize_message

logger = init_logger(__name__)


def create_error_response(
    message: str | Exception,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    param: str | None = None,
) -> ErrorResponse:
    exc: Exception | None = None
    message_text = str(message)

    if isinstance(message, Exception):
        exc = message
        logger.debug("create_error_response called with %s: %s", type(exc).__name__, exc)

        from aphrodite.exceptions import (
            AphroditeClientError,
            AphroditeNotFoundError,
            AphroditeServerError,
            AphroditeUnprocessableEntityError,
            AphroditeValidationError,
            GracefulHTTPError,
        )
        from aphrodite.v1.engine.exceptions import EngineDeadError

        if isinstance(exc, GracefulHTTPError):
            err_type = HTTPStatus(exc.http_status).phrase
            status_code = exc.http_status
            param = None
        elif isinstance(exc, AphroditeValidationError):
            err_type = "BadRequestError"
            status_code = HTTPStatus.BAD_REQUEST
            param = exc.parameter
        elif isinstance(exc, AphroditeUnprocessableEntityError):
            err_type = "UnprocessableEntityError"
            status_code = HTTPStatus.UNPROCESSABLE_ENTITY
            param = exc.parameter
        elif isinstance(exc, AphroditeNotFoundError):
            err_type = "NotFoundError"
            status_code = HTTPStatus.NOT_FOUND
            param = None
        elif isinstance(exc, AphroditeClientError):
            err_type = "BadRequestError"
            status_code = HTTPStatus.BAD_REQUEST
            param = None
        elif isinstance(exc, GenerationError):
            err_type = "ServiceUnavailableError" if isinstance(exc, RetryableRequestError) else "InternalServerError"
            status_code = exc.status_code
            param = None
        elif isinstance(exc, EngineDeadError):
            err_type = "InternalServerError"
            status_code = HTTPStatus.INTERNAL_SERVER_ERROR
            param = None
            message_text = exc.CLIENT_MESSAGE
        elif isinstance(exc, AphroditeServerError):
            err_type = "InternalServerError"
            status_code = HTTPStatus.INTERNAL_SERVER_ERROR
            param = None
        # Fallback for raw exceptions not yet migrated to AphroditeError.
        elif isinstance(exc, (ValueError, TypeError, OverflowError)):
            err_type = "BadRequestError"
            status_code = HTTPStatus.BAD_REQUEST
            param = None
        elif isinstance(exc, NotImplementedError):
            err_type = "NotImplementedError"
            status_code = HTTPStatus.NOT_IMPLEMENTED
            param = None
        elif any(cls.__name__ == "TemplateError" for cls in type(exc).__mro__):
            # jinja2.TemplateError and its subclasses (avoid importing jinja2)
            err_type = "BadRequestError"
            status_code = HTTPStatus.BAD_REQUEST
            param = None
        else:
            err_type = "InternalServerError"
            status_code = HTTPStatus.INTERNAL_SERVER_ERROR
            param = None

    return ErrorResponse(
        error=ErrorInfo(
            message=sanitize_message(message_text),
            type=err_type,
            code=status_code.value,
            param=param,
        )
    )
