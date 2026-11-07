# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for unprocessable media URL error handling."""

from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from sonar.connections import (
    MediaDownloadSizeExceededError,
    MediaRequestDownloadSizeExceededError,
)
from sonar.entrypoints.serve import create_error_response
from sonar.exceptions import SonarUnprocessableEntityError
from sonar.multimodal.media import MediaConnector

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class TestSonarUnprocessableEntityError:
    def test_creation(self):
        exc = SonarUnprocessableEntityError("Test error")
        assert str(exc) == "Test error"
        assert exc.parameter is None

    def test_creation_with_parameter_and_value(self):
        exc = SonarUnprocessableEntityError(
            "Test error",
            parameter="image_url",
            value="https://example.com/image.jpg",
        )
        assert "parameter=image_url" in str(exc)
        assert "value=https://example.com/image.jpg" in str(exc)

    def test_is_value_error_subclass(self):
        exc = SonarUnprocessableEntityError("Test")
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

            with pytest.raises(SonarUnprocessableEntityError) as exc_info:
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

            with pytest.raises(SonarUnprocessableEntityError) as exc_info:
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

    def test_fetch_image_download_size_limit_error(self):
        connector = MediaConnector()

        with patch.object(connector.connection, "get_bytes", new_callable=MagicMock) as mock_get:
            mock_get.side_effect = MediaDownloadSizeExceededError(256 * 1024 * 1024)

            with pytest.raises(SonarUnprocessableEntityError) as exc_info:
                connector.fetch_image("https://example.com/image.jpg")

            assert "SONAR_MAX_MEDIA_DOWNLOAD_SIZE_MB=256" in str(exc_info.value)

    def test_fetch_image_request_download_size_limit_error(self):
        connector = MediaConnector()

        with patch.object(connector.connection, "get_bytes", new_callable=MagicMock) as mock_get:
            mock_get.side_effect = MediaRequestDownloadSizeExceededError("SONAR_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB=256")

            with pytest.raises(SonarUnprocessableEntityError) as exc_info:
                connector.fetch_image("https://example.com/image.jpg")

            assert "SONAR_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB=256" in str(exc_info.value)


class TestErrorResponse:
    def test_unprocessable_entity_returns_422(self):
        exc = SonarUnprocessableEntityError(
            "Failed to fetch media from URL: Cannot connect",
            parameter="image_url",
            value="https://example.com/image.jpg",
        )

        response = create_error_response(exc)

        assert response.error.code == HTTPStatus.UNPROCESSABLE_ENTITY.value
        assert response.error.type == "UnprocessableEntityError"
        assert response.error.param == "image_url"

    def test_unprocessable_entity_message(self):
        exc = SonarUnprocessableEntityError("Test error message")
        response = create_error_response(exc)

        assert response.error.message == "Test error message"
        assert response.error.code == 422
