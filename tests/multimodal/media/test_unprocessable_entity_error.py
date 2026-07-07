# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for unprocessable media URL error handling."""

from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from aphrodite.entrypoints.serve.utils.error_response import create_error_response
from aphrodite.exceptions import APHRODITEUnprocessableEntityError
from aphrodite.multimodal.media import MediaConnector


class TestAphroditeUnprocessableEntityError:
    def test_creation(self):
        exc = APHRODITEUnprocessableEntityError("Test error")
        assert str(exc) == "Test error"
        assert exc.parameter is None

    def test_creation_with_parameter_and_value(self):
        exc = APHRODITEUnprocessableEntityError(
            "Test error",
            parameter="image_url",
            value="https://example.com/image.jpg",
        )
        assert "parameter=image_url" in str(exc)
        assert "value=https://example.com/image.jpg" in str(exc)

    def test_is_value_error_subclass(self):
        exc = APHRODITEUnprocessableEntityError("Test")
        assert isinstance(exc, ValueError)


class TestMediaConnectorErrorHandling:
    @pytest.mark.asyncio
    async def test_fetch_image_async_404(self):
        connector = MediaConnector()

        with patch.object(
            connector.connection,
            "async_get_bytes",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=404,
                message="Not Found",
            )

            with pytest.raises(APHRODITEUnprocessableEntityError) as exc_info:
                await connector.fetch_image_async("https://example.com/missing.jpg")

            assert exc_info.value.parameter == "image_url"

    @pytest.mark.asyncio
    async def test_fetch_image_async_dns_error(self):
        connector = MediaConnector()

        with patch.object(
            connector.connection,
            "async_get_bytes",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientConnectionError("DNS lookup failed")

            with pytest.raises(aiohttp.ClientConnectionError) as exc_info:
                await connector.fetch_image_async("https://nonexistent.example/image.jpg")

            assert isinstance(exc_info.value, aiohttp.ClientConnectionError)

    @pytest.mark.asyncio
    async def test_fetch_image_async_500_preserved(self):
        connector = MediaConnector()

        with patch.object(
            connector.connection,
            "async_get_bytes",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=500,
                message="Internal Server Error",
            )

            with pytest.raises(aiohttp.ClientResponseError) as exc_info:
                await connector.fetch_image_async("https://example.com/image.jpg")

            assert exc_info.value.status == 500

    def test_fetch_image_404(self):
        connector = MediaConnector()

        with patch.object(
            connector.connection,
            "get_bytes",
            new_callable=MagicMock,
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=404,
                message="Not Found",
            )

            with pytest.raises(APHRODITEUnprocessableEntityError) as exc_info:
                connector.fetch_image("https://example.com/missing.jpg")

            assert exc_info.value.parameter == "image_url"

    def test_fetch_image_connection_error(self):
        connector = MediaConnector()

        with patch.object(
            connector.connection,
            "get_bytes",
            new_callable=MagicMock,
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientConnectionError("Connection refused")

            with pytest.raises(aiohttp.ClientConnectionError) as exc_info:
                connector.fetch_image("https://example.com/image.jpg")

            assert isinstance(exc_info.value, aiohttp.ClientConnectionError)


class TestErrorResponse:
    def test_unprocessable_entity_returns_422(self):
        exc = APHRODITEUnprocessableEntityError(
            "Failed to fetch media from URL: Cannot connect",
            parameter="image_url",
            value="https://example.com/image.jpg",
        )

        response = create_error_response(exc)

        assert response.error.code == HTTPStatus.UNPROCESSABLE_ENTITY.value
        assert response.error.type == "UnprocessableEntityError"
        assert response.error.param == "image_url"

    def test_unprocessable_entity_message(self):
        exc = APHRODITEUnprocessableEntityError("Test error message")
        response = create_error_response(exc)

        assert response.error.message == "Test error message"
        assert response.error.code == 422
