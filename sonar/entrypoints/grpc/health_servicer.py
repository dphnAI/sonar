# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
# Adapted from smg-grpc-servicer 0.5.2; the external VllmEngine wire identity is retained.
"""
Standard gRPC health check service for Aphrodite Kubernetes probes.

Implements grpc.health.v1.Health protocol, delegating health status
to AsyncLLM.check_health() from the Aphrodite EngineClient protocol.
"""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

from aphrodite.logger import init_logger

if TYPE_CHECKING:
    from aphrodite.v1.engine.async_llm import AsyncLLM

logger = init_logger(__name__)


class AphroditeHealthServicer(health_pb2_grpc.HealthServicer):
    """
    Standard gRPC health check service for Kubernetes probes.
    Implements grpc.health.v1.Health protocol.

    Supports two service levels:
    1. Overall server health (service="") - for liveness probes
    2. VllmEngine service health (service="vllm.grpc.engine.VllmEngine")
       - for readiness probes

    Health is determined by calling async_llm.check_health(), the same
    EngineClient protocol method used by Aphrodite's HTTP /health endpoint.
    """

    OVERALL_SERVER = ""
    VLLM_SERVICE = "vllm.grpc.engine.VllmEngine"

    def __init__(self, async_llm: "AsyncLLM"):
        """
        Initialize health servicer.

        Args:
            async_llm: AsyncLLM instance for checking engine health
        """
        self.async_llm = async_llm
        self._shutting_down = False
        logger.info("Standard gRPC health service initialized")

    def set_not_serving(self):
        """Mark all services as NOT_SERVING during graceful shutdown."""
        self._shutting_down = True
        logger.info("Health service status set to NOT_SERVING")

    async def Check(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> health_pb2.HealthCheckResponse:
        """
        Standard health check for Kubernetes probes.

        Args:
            request: Contains service name ("" for overall, or specific service)
            context: gRPC context

        Returns:
            HealthCheckResponse with SERVING/NOT_SERVING/SERVICE_UNKNOWN status
        """
        service_name = request.service
        logger.debug("Health check request for service: '%s'", service_name)

        if self._shutting_down:
            logger.debug("Health check: Server is shutting down")
            return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.NOT_SERVING)

        if service_name in (self.OVERALL_SERVER, self.VLLM_SERVICE):
            try:
                await self.async_llm.check_health()
                logger.debug("Health check for '%s': SERVING", service_name)
                return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVING)
            except Exception:
                logger.exception("Health check failed for service '%s'", service_name)
                return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.NOT_SERVING)

        logger.debug("Health check for unknown service: '%s'", service_name)
        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details(f"Unknown service: {service_name}")
        return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVICE_UNKNOWN)

    async def Watch(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[health_pb2.HealthCheckResponse]:
        """
        Streaming health check - sends current status once.

        For now, sends current status once (Kubernetes doesn't use Watch).
        A full implementation would monitor status changes and stream updates.

        Args:
            request: Contains service name
            context: gRPC context

        Yields:
            HealthCheckResponse messages
        """
        service_name = request.service
        logger.debug("Health watch request for service: '%s'", service_name)

        # Inline status computation to avoid Check()'s context.set_code()
        # side effect, which would incorrectly set the RPC status on the
        # streaming response for unknown services.
        status = health_pb2.HealthCheckResponse.SERVICE_UNKNOWN
        if self._shutting_down:
            status = health_pb2.HealthCheckResponse.NOT_SERVING
        elif service_name in (self.OVERALL_SERVER, self.VLLM_SERVICE):
            try:
                await self.async_llm.check_health()
                status = health_pb2.HealthCheckResponse.SERVING
            except Exception:
                logger.debug(
                    "Health watch check failed for service '%s'",
                    service_name,
                    exc_info=True,
                )
                status = health_pb2.HealthCheckResponse.NOT_SERVING

        yield health_pb2.HealthCheckResponse(status=status)
