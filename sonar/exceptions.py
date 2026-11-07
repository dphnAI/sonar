# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Custom exceptions for Sonar."""

from http import HTTPStatus
from typing import Any


class SonarError(Exception):
    """Base class for all Sonar-specific errors."""


class SonarClientError(SonarError):
    """Base class for errors caused by the client request (4xx)."""


class SonarServerError(SonarError):
    """Base class for errors caused by the server (5xx)."""


class SonarValidationError(SonarClientError):
    """Sonar-specific validation error for request validation failures.

    Args:
        message: The error message describing the validation failure.
        parameter: Optional parameter name that failed validation.
        value: Optional value that was rejected during validation.

    """

    def __init__(
        self,
        message: str,
        *,
        parameter: str | None = None,
        value: Any = None,
    ) -> None:
        super().__init__(message)
        self.parameter = parameter
        self.value = value

    def __str__(self):
        base = super().__str__()
        extras = []
        if self.parameter is not None:
            extras.append(f"parameter={self.parameter}")
        if self.value is not None:
            extras.append(f"value={self.value}")
        return f"{base} ({', '.join(extras)})" if extras else base


class SonarNotFoundError(SonarClientError):
    """Sonar-specific NotFoundError."""

    pass


class ChatTemplateResolutionError(SonarClientError):
    """Raised when chat template resolution fails."""

    pass


class LoRAAdapterNotFoundError(SonarNotFoundError):
    """Exception raised when a LoRA adapter is not found.

    This exception is thrown when a requested LoRA adapter does not exist
    in the system.

    Attributes:
        message: The error message string describing the exception

    """

    message: str

    def __init__(
        self,
        lora_name: str,
        lora_path: str,
    ) -> None:
        message = f"Loading lora {lora_name} failed: No adapter found for {lora_path}"
        self.message = message

    def __str__(self):
        return self.message


class SonarUnprocessableEntityError(SonarClientError):
    """Sonar-specific error for unprocessable entity requests.

    Args:
        message: The error message describing the unprocessable entity.
        parameter: Optional parameter name that failed validation.
        value: Optional value that was rejected during validation.

    """

    def __init__(
        self,
        message: str,
        *,
        parameter: str | None = None,
        value: Any = None,
    ) -> None:
        super().__init__(message)
        self.parameter = parameter
        self.value = value

    def __str__(self):
        base = super().__str__()
        extras = []
        if self.parameter is not None:
            extras.append(f"parameter={self.parameter}")
        if self.value is not None:
            extras.append(f"value={self.value}")
        return f"{base} ({', '.join(extras)})" if extras else base


class GenerationError(SonarServerError):
    """raised when finish_reason indicates internal server error (500)"""

    def __init__(self, message: str = "Internal server error"):
        super().__init__(message)
        self.status_code = HTTPStatus.INTERNAL_SERVER_ERROR


class RetryableRequestError(GenerationError):
    """A transient request-level failure that is safe for clients to retry."""

    def __init__(self, message: str = "Request failed; please retry"):
        super().__init__(message)
        self.status_code = HTTPStatus.SERVICE_UNAVAILABLE


class GracefulHTTPError(SonarError):
    """Exception that should be translated into an HTTP error response.

    These are expected to occur during normal operation (e.g. admission
    control rejections) and should be surfaced to the client with the
    explicit HTTP status code they carry, rather than being mapped to a
    generic 4xx/5xx by the client/server split.
    """

    def __init__(self, message: str, http_status: HTTPStatus):
        super().__init__(message)
        self.message = message
        self.http_status = http_status


class QueueOverflowError(GracefulHTTPError):
    """Raised when admitting a request would exceed the request queue limit.

    Returns HTTP 503 (Service Unavailable) so that load balancers and
    client SDKs retry the request on a different instance.
    """

    def __init__(self):
        super().__init__(
            "The engine is currently busy and cannot accept new requests. "
            "Please try again later or on a different instance.",
            HTTPStatus.SERVICE_UNAVAILABLE,
        )


class MaxQueuedTokensError(GracefulHTTPError):
    """Raised when the pending prefill tokens exceed the configured limit.

    Returns HTTP 503 (Service Unavailable) so that load balancers and
    client SDKs retry the request on a different instance.
    """

    def __init__(self):
        super().__init__(
            "The engine has reached its prefill token backlog limit. "
            "Please try again later or on a different instance.",
            HTTPStatus.SERVICE_UNAVAILABLE,
        )


class ProfilerAlreadyActiveError(GracefulHTTPError):
    """Raised when a profiling session is already active."""

    def __init__(self):
        super().__init__(
            "A profiling session is already active. Call /stop_profile before starting another session.",
            HTTPStatus.CONFLICT,
        )


class EngineFaultedError(GracefulHTTPError):
    """Raised when the engine has faulted and is awaiting FT recovery.

    Returns HTTP 503 (Service Unavailable) so that load balancers and
    client SDKs retry the request on a different instance.
    """

    def __init__(self):
        super().__init__(
            "The engine has faulted and cannot accept new requests. Please try again later or on a different instance.",
            HTTPStatus.SERVICE_UNAVAILABLE,
        )
