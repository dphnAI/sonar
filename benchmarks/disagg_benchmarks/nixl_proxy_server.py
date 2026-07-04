# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import asyncio
import logging
import os
import uuid

import aiohttp
from quart import Quart, Response, make_response, request
from rate_limiter import RateLimiter
from request_queue import RequestQueue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Aphrodite NIXL disaggregation proxy server")
    parser.add_argument("--timeout", type=float, default=300, help="Timeout for backend service requests in seconds")
    parser.add_argument(
        "--max-concurrent", type=int, default=100, help="Maximum concurrent requests to backend services"
    )
    parser.add_argument("--queue-size", type=int, default=500, help="Maximum number of requests in the queue")
    parser.add_argument("--rate-limit", type=int, default=40, help="Maximum requests per second")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument(
        "--prefill-url", type=str, default="http://localhost:8100/v1/completions", help="Prefill service endpoint URL"
    )
    parser.add_argument(
        "--decode-url", type=str, default="http://localhost:8200/v1/completions", help="Decode service endpoint URL"
    )
    parser.add_argument("--prefill-engine-id", type=str, default="prefill_engine", help="Prefill engine ID")
    parser.add_argument("--decode-engine-id", type=str, default="decode_engine", help="Decode engine ID")
    parser.add_argument("--prefill-nixl-host", type=str, default="127.0.0.1", help="Prefill NIXL side-channel host")
    parser.add_argument("--prefill-nixl-port", type=int, default=14590, help="Prefill NIXL side-channel port")
    parser.add_argument("--decode-nixl-host", type=str, default="127.0.0.1", help="Decode NIXL side-channel host")
    parser.add_argument("--decode-nixl-port", type=int, default=14591, help="Decode NIXL side-channel port")
    return parser.parse_args()


def main():
    args = parse_args()

    AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=args.timeout)
    MAX_CONCURRENT_REQUESTS = args.max_concurrent
    REQUEST_QUEUE_SIZE = args.queue_size
    RATE_LIMIT = args.rate_limit
    PREFILL_SERVICE_URL = args.prefill_url
    DECODE_SERVICE_URL = args.decode_url
    PREFILL_ENGINE_ID = args.prefill_engine_id
    DECODE_ENGINE_ID = args.decode_engine_id
    PREFILL_NIXL_HOST = args.prefill_nixl_host
    PREFILL_NIXL_PORT = args.prefill_nixl_port
    DECODE_NIXL_HOST = args.decode_nixl_host
    DECODE_NIXL_PORT = args.decode_nixl_port
    PORT = args.port

    app = Quart(__name__)

    rate_limiter = RateLimiter(RATE_LIMIT)
    request_queue = RequestQueue(MAX_CONCURRENT_REQUESTS, REQUEST_QUEUE_SIZE)

    app.config.update(
        {
            "AIOHTTP_TIMEOUT": AIOHTTP_TIMEOUT,
            "rate_limiter": rate_limiter,
            "request_queue": request_queue,
            "PREFILL_SERVICE_URL": PREFILL_SERVICE_URL,
            "DECODE_SERVICE_URL": DECODE_SERVICE_URL,
            "PREFILL_ENGINE_ID": PREFILL_ENGINE_ID,
            "DECODE_ENGINE_ID": DECODE_ENGINE_ID,
            "PREFILL_NIXL_HOST": PREFILL_NIXL_HOST,
            "PREFILL_NIXL_PORT": PREFILL_NIXL_PORT,
            "DECODE_NIXL_HOST": DECODE_NIXL_HOST,
            "DECODE_NIXL_PORT": DECODE_NIXL_PORT,
        }
    )

    @app.before_serving
    async def startup():
        asyncio.create_task(request_queue.process())

    def random_uuid() -> str:
        return str(uuid.uuid4().hex)

    async def forward_request(url, data, request_id, kv_transfer_params=None):
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }

        # Add NIXL metadata to request body if provided
        request_data = data.copy()
        if kv_transfer_params:
            request_data["kv_transfer_params"] = kv_transfer_params

        async with (
            rate_limiter,
            aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session,
        ):
            try:
                async with session.post(url=url, json=request_data, headers=headers) as response:
                    if response.status == 200:
                        async for chunk_bytes in response.content.iter_chunked(1024):
                            yield chunk_bytes
                    else:
                        error_text = await response.text()
                        logger.error(
                            "Backend request failed with status %d: %s",
                            response.status,
                            error_text,
                        )
                        yield b'{"error": "Backend service error"}\n'
            except asyncio.TimeoutError:
                logger.error("Backend request timed out")
                yield b'{"error": "Backend request timed out"}\n'
            except Exception as e:
                logger.exception("Backend request failed: %s", e)
                yield b'{"error": "Backend request failed"}\n'

    async def process_request():
        try:
            original_request_data = await request.get_json()

            # Create unique request ID
            request_id = f"nixl_{random_uuid()}"

            prompt = original_request_data.get("prompt", "")
            prompt_preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
            max_tokens = original_request_data.get("max_tokens", 0)

            logger.info("=" * 80)
            logger.info("📥 NEW REQUEST: %s", request_id)
            logger.info("   Prompt: %s", prompt_preview)
            logger.info("   Max tokens: %d", max_tokens)
            logger.info("-" * 80)

            # Create prefill request (max_tokens=1 to just do prefill)
            prefill_request = original_request_data.copy()
            prefill_request["max_tokens"] = 1

            # Execute prefill stage
            logger.info("⚡ PREFILL: Sending to %s", PREFILL_SERVICE_URL)
            async for _ in forward_request(PREFILL_SERVICE_URL, prefill_request, request_id):
                continue
            logger.info("✓ PREFILL: Complete")

            # Prepare NIXL metadata for decode instance
            # This tells decode where to pull KV cache from
            kv_transfer_params = {
                "remote_engine_id": PREFILL_ENGINE_ID,
                "remote_host": PREFILL_NIXL_HOST,
                "remote_port": PREFILL_NIXL_PORT,
                # Note: remote_block_ids will be determined by decode based on request
            }

            # Execute decode stage with NIXL metadata
            logger.info("🔄 DECODE: Sending to %s with NIXL metadata", DECODE_SERVICE_URL)
            logger.info("   Pulling KV from: %s (%s:%d)", PREFILL_ENGINE_ID, PREFILL_NIXL_HOST, PREFILL_NIXL_PORT)
            generator = forward_request(
                DECODE_SERVICE_URL, original_request_data, request_id, kv_transfer_params=kv_transfer_params
            )
            response = await make_response(generator)
            response.headers["Content-Type"] = "application/json"
            response.timeout = None
            logger.info("✓ DECODE: Complete")
            logger.info("=" * 80)
            return response

        except Exception:
            logger.exception("Error processing request")
            return Response(
                response=b'{"error": "Internal server error"}',
                status=500,
                content_type="application/json",
            )

    @app.route("/v1/completions", methods=["POST"])
    async def handle_request():
        task = asyncio.create_task(process_request())

        if not await request_queue.enqueue(task):
            return Response(
                response=b'{"error": "Server busy, try again later"}',
                status=503,
                content_type="application/json",
            )

        try:
            return await task
        except asyncio.CancelledError:
            logger.warning("Request cancelled due to timeout or queue full")
            return Response(
                response=b'{"error": "Request cancelled"}',
                status=503,
                content_type="application/json",
            )

    @app.route("/v1/models", methods=["GET"])
    async def handle_models():
        """Forward /v1/models requests to prefill instance."""
        try:
            async with (
                aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session,
                session.get(url=PREFILL_SERVICE_URL.replace("/v1/completions", "/v1/models")) as response,
            ):
                content = await response.read()
                return Response(
                    response=content,
                    status=response.status,
                    content_type=response.content_type,
                )
        except Exception as e:
            logger.error("Error forwarding /v1/models request: %s", str(e))
            return Response(
                response=b'{"error": "Service unavailable"}',
                status=503,
                content_type="application/json",
            )

    @app.route("/v1/tokenize", methods=["POST"])
    async def handle_tokenize():
        """Forward /v1/tokenize requests to prefill instance."""
        try:
            request_data = await request.get_json()
            async with (
                aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session,
                session.post(
                    url=PREFILL_SERVICE_URL.replace("/v1/completions", "/v1/tokenize"),
                    json=request_data,
                ) as response,
            ):
                content = await response.read()
                return Response(
                    response=content,
                    status=response.status,
                    content_type=response.content_type,
                )
        except Exception as e:
            logger.error("Error forwarding /v1/tokenize request: %s", str(e))
            return Response(
                response=b'{"error": "Service unavailable"}',
                status=503,
                content_type="application/json",
            )

    @app.route("/health", methods=["GET"])
    async def health():
        return Response(response=b'{"status": "ok"}', status=200, content_type="application/json")

    logger.info("Starting NIXL proxy server on port %d", PORT)
    logger.info(
        "Prefill: %s (engine_id=%s, nixl=%s:%d)",
        PREFILL_SERVICE_URL,
        PREFILL_ENGINE_ID,
        PREFILL_NIXL_HOST,
        PREFILL_NIXL_PORT,
    )
    logger.info(
        "Decode: %s (engine_id=%s, nixl=%s:%d)",
        DECODE_SERVICE_URL,
        DECODE_ENGINE_ID,
        DECODE_NIXL_HOST,
        DECODE_NIXL_PORT,
    )

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, backlog=8192, log_level="info")


if __name__ == "__main__":
    main()
