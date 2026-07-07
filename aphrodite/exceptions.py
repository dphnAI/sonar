# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Custom exceptions for Aphrodite."""

from typing import Any


class AphroditeValidationError(ValueError):
    """Aphrodite-specific validation error for request validation failures.

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


class AphroditeNotFoundError(Exception):
    """Aphrodite-specific NotFoundError"""

    pass


class LoRAAdapterNotFoundError(AphroditeNotFoundError):
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


class AphroditeUnprocessableEntityError(ValueError):
    """Aphrodite-specific error for unprocessable entity requests.

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


# Backward compatibility with older upstream-derived imports.
APHRODITEValidationError = AphroditeValidationError
APHRODITENotFoundError = AphroditeNotFoundError
APHRODITEUnprocessableEntityError = AphroditeUnprocessableEntityError
