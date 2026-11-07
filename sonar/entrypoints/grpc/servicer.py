# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
# Adapted from smg-grpc-servicer 0.5.2; the external VllmEngine wire identity is retained.
"""
Aphrodite gRPC Servicer

Implements the VllmEngine gRPC service on top of Aphrodite's EngineClient.
"""

import itertools
import time
from collections.abc import AsyncGenerator, Sequence

import grpc
import torch
from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc
from transformers import BatchFeature

from aphrodite import PoolingParams, SamplingParams, TokensPrompt
from aphrodite.config.model import get_served_model_name
from aphrodite.engine.protocol import EngineClient
from aphrodite.exceptions import AphroditeValidationError
from aphrodite.inputs.engine import MultiModalInput as AphroditeMultiModalInput
from aphrodite.inputs.engine import mm_input, tokens_input
from aphrodite.logger import init_logger
from aphrodite.logprobs import Logprob, PromptLogprobs, SampleLogprobs
from aphrodite.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from aphrodite.outputs import CompletionOutput, RequestOutput
from aphrodite.sampling_params import RequestOutputKind, StructuredOutputsParams

logger = init_logger(__name__)

# Proto dtype string → torch dtype
_PROTO_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "int64": torch.int64,
    "uint32": torch.uint32,
}


def _tensor_from_proto(td: vllm_engine_pb2.TensorData) -> torch.Tensor:
    """Deserialize a TensorData proto message into a torch.Tensor."""
    torch_dtype = _PROTO_DTYPE_MAP.get(td.dtype)
    if torch_dtype is None:
        raise ValueError(f"Unsupported proto tensor dtype: {td.dtype!r}")
    return torch.frombuffer(bytearray(td.data), dtype=torch_dtype).reshape(*td.shape)


class AphroditeEngineServicer(vllm_engine_pb2_grpc.VllmEngineServicer):
    """
    gRPC servicer implementing the VllmEngine service.

    Handles 6 RPCs:
    - Generate: Streaming text generation
    - Embed: Embeddings
    - HealthCheck: Health probe
    - Abort: Cancel requests out-of-band
    - GetModelInfo: Model metadata
    - GetServerInfo: Server state
    """

    def __init__(self, async_llm: EngineClient, start_time: float):
        """
        Initialize the servicer.

        Args:
            async_llm: The EngineClient instance (e.g. AsyncLLM)
            start_time: The server start time, in seconds since epoch
        """
        self.engine = async_llm
        self.start_time = start_time
        logger.info("AphroditeEngineServicer initialized")

    async def Generate(
        self,
        request: vllm_engine_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[vllm_engine_pb2.GenerateResponse, None]:
        """
        Handle streaming generation requests.

        Supports n>1 by sending separate chunk/complete messages for each output index.
        When streaming with n>1, chunks for different indices are interleaved.

        Args:
            request: The GenerateRequest protobuf
            context: gRPC context

        Yields:
            GenerateResponse protobuf messages (streaming)
        """
        request_id = request.request_id
        input_type = request.WhichOneof("input")
        has_preprocessed_mm = request.HasField("mm_inputs") and request.mm_inputs.HasField("pixel_values")
        logger.info(
            "Generate request %s: input_type=%s, stream=%s, preprocessed_mm=%s",
            request_id,
            input_type,
            request.stream,
            has_preprocessed_mm,
        )

        try:
            arrival_time = time.time()
            prompt: AphroditeMultiModalInput | TokensPrompt | str

            if has_preprocessed_mm and input_type == "tokenized":
                # Preprocessed multimodal from Rust router.
                # Token IDs already have expanded placeholders; tensors are
                # ready for the model. Bypass the renderer entirely.
                prompt = self._build_preprocessed_mm_inputs(request.tokenized, request.mm_inputs)
                prompt["arrival_time"] = arrival_time
            elif input_type == "tokenized":
                prompt = {"prompt_token_ids": list(request.tokenized.input_ids)}
                if request.tokenized.original_text:
                    prompt["prompt"] = request.tokenized.original_text
                # Keep this as a raw TokensPrompt so InputProcessor applies
                # tokenization_kwargs (including truncation) before rendering.
            else:
                prompt = request.text

            # Build sampling params with detokenize=False
            sampling_params = self._sampling_params_from_proto(
                request.sampling_params,
                stream=request.stream,
                kv_transfer_params=request.kv_transfer_params if request.HasField("kv_transfer_params") else None,
            )
            tokenization_kwargs = self._tokenization_kwargs_from_proto(request.sampling_params)

            # Extract logprobs configuration
            num_logprobs = sampling_params.logprobs
            num_prompt_logprobs = sampling_params.prompt_logprobs

            # Track which indices have sent their first chunk
            seen_indices: set[int] = set()

            async for output in self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                tokenization_kwargs=tokenization_kwargs,
            ):
                # For streaming, send chunks for EACH completion output (n outputs)
                if request.stream:
                    for completion in output.outputs:
                        idx = completion.index
                        is_first = idx not in seen_indices
                        seen_indices.add(idx)

                        # Send chunk with delta data (Rust accumulates for Aphrodite)
                        yield self._chunk_response(
                            output,
                            completion=completion,
                            num_logprobs=num_logprobs,
                            num_prompt_logprobs=num_prompt_logprobs,
                            is_first_chunk=is_first,
                        )

                        # Send Complete when sequence finishes (n>1 support)
                        if completion.finish_reason:
                            yield self._complete_response(
                                output,
                                completion=completion,
                                num_logprobs=num_logprobs,
                                num_prompt_logprobs=num_prompt_logprobs,
                            )

                # For non-streaming, send complete response when finished
                if output.finished and not request.stream:
                    for completion in output.outputs:
                        yield self._complete_response(
                            output,
                            completion=completion,
                            num_logprobs=num_logprobs,
                            num_prompt_logprobs=num_prompt_logprobs,
                        )

        except (ValueError, AphroditeValidationError) as e:
            # Invalid request error (equiv to 400).
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            logger.exception("Error in Generate for request %s", request_id)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def Embed(
        self,
        request: vllm_engine_pb2.EmbedRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.EmbedResponse:
        """
        Handle embedding requests.

        Calls Aphrodite's encode() API with PoolingParams and returns the embedding vector.

        Args:
            request: The EmbedRequest protobuf
            context: gRPC context

        Returns:
            EmbedResponse protobuf
        """
        request_id = request.request_id
        logger.info("Embed request %s", request_id)

        try:
            if not request.HasField("tokenized"):
                raise ValueError("EmbedRequest requires tokenized input")

            prompt = tokens_input(
                prompt_token_ids=list(request.tokenized.input_ids),
                prompt=request.tokenized.original_text or None,
            )

            pooling_params = PoolingParams(task="embed")

            # encode() is an async generator; collect the final result
            final_output = None
            async for output in self.engine.encode(
                prompt=prompt,
                pooling_params=pooling_params,
                request_id=request_id,
            ):
                final_output = output

            if final_output is not None and final_output.finished:
                embedding = final_output.outputs.data.tolist()
                return vllm_engine_pb2.EmbedResponse(
                    embedding=embedding,
                    prompt_tokens=len(final_output.prompt_token_ids),
                    embedding_dim=len(embedding),
                )

            msg = f"Embed request {request_id} did not produce a result"
            logger.warning(msg)
            await context.abort(grpc.StatusCode.INTERNAL, msg)

        except grpc.aio.AbortError:
            raise
        except (ValueError, AphroditeValidationError) as e:
            logger.warning("Embed invalid request %s: %s", request_id, e)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            logger.exception("Embed failed for request %s", request_id)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def HealthCheck(
        self,
        request: vllm_engine_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.HealthCheckResponse:
        """
        Handle health check requests.

        Args:
            request: The HealthCheckRequest protobuf
            context: gRPC context

        Returns:
            HealthCheckResponse protobuf
        """
        is_healthy = not self.engine.errored
        message = "Health" if is_healthy else "Engine is not alive"

        logger.info("HealthCheck request: healthy=%s, message=%s", is_healthy, message)

        return vllm_engine_pb2.HealthCheckResponse(healthy=is_healthy, message=message)

    async def Abort(
        self,
        request: vllm_engine_pb2.AbortRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.AbortResponse:
        """
        Out-of-band abort requests.

        Args:
            request: The AbortRequest protobuf
            context: gRPC context

        Returns:
            AbortResponse protobuf
        """
        request_ids = request.request_ids
        logger.info("Abort requests: %s", request_ids)

        await self.engine.abort(request_ids)
        return vllm_engine_pb2.AbortResponse()

    async def GetModelInfo(
        self,
        request: vllm_engine_pb2.GetModelInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.GetModelInfoResponse:
        """
        Handle model info requests.

        Args:
            request: The GetModelInfoRequest protobuf
            context: gRPC context

        Returns:
            GetModelInfoResponse protobuf
        """
        model_config = self.engine.model_config
        hf_config = model_config.hf_config

        # eos_token_id can be int or list[int]
        eos = getattr(hf_config, "eos_token_id", None)
        if isinstance(eos, int):
            eos_token_ids = [eos]
        elif isinstance(eos, list):
            eos_token_ids = eos
        else:
            eos_token_ids = []

        return vllm_engine_pb2.GetModelInfoResponse(
            model_path=model_config.model,
            is_generation=model_config.runner_type == "generate",
            max_context_length=model_config.max_model_len,
            vocab_size=model_config.get_vocab_size(),
            supports_vision=model_config.is_multimodal_model,
            served_model_name=get_served_model_name(model_config.model, model_config.served_model_name),
            tokenizer_path=model_config.tokenizer or "",
            model_type=getattr(hf_config, "model_type", "") or "",
            architectures=model_config.architectures or [],
            eos_token_ids=eos_token_ids,
            pad_token_id=getattr(hf_config, "pad_token_id", None) or 0,
            bos_token_id=getattr(hf_config, "bos_token_id", None) or 0,
            max_req_input_len=model_config.max_model_len,
        )

    async def GetServerInfo(
        self,
        request: vllm_engine_pb2.GetServerInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.GetServerInfoResponse:
        """
        Handle server info requests.

        Args:
            request: The GetServerInfoRequest protobuf
            context: gRPC context

        Returns:
            GetServerInfoResponse protobuf
        """
        kv_connector = ""
        kv_role = ""
        kv_transfer_config = self.engine.aphrodite_config.kv_transfer_config
        if kv_transfer_config is not None:
            kv_connector = kv_transfer_config.kv_connector or ""
            kv_role = kv_transfer_config.kv_role or ""

        return vllm_engine_pb2.GetServerInfoResponse(
            kv_connector=kv_connector,
            kv_role=kv_role,
        )

    # ========== Helper methods ==========

    def _build_preprocessed_mm_inputs(
        self,
        tokenized: vllm_engine_pb2.TokenizedInput,
        mm_proto: vllm_engine_pb2.MultimodalInputs,
    ) -> AphroditeMultiModalInput:
        """Build Aphrodite MultiModalInput from preprocessed proto data.

        Bypasses HF processor entirely — pixel values and model-specific
        tensors were already computed by the Rust router.  Field layouts
        (batched / flat / shared) are also determined by the router via
        ``batched_keys`` and ``flat_keys`` proto fields.
        """
        prompt_token_ids = list(tokenized.input_ids)
        num_images = len(mm_proto.mm_placeholders)

        # Deserialize all tensors from proto
        hf_dict: dict[str, torch.Tensor] = {
            "pixel_values": _tensor_from_proto(mm_proto.pixel_values),
        }
        for key, td in mm_proto.model_specific_tensors.items():
            hf_dict[key] = _tensor_from_proto(td)

        # Cast floating-point tensors to model dtype (e.g. bfloat16).
        # This mirrors _postprocess_output in multimodal/processing/context.py
        # which is skipped when bypassing the HF processor.
        model_dtype = self.engine.model_config.dtype
        for key in hf_dict:
            if hf_dict[key].is_floating_point():
                hf_dict[key] = hf_dict[key].to(dtype=model_dtype)

        cpu_keys = set(mm_proto.keep_on_cpu_keys)

        # Field configs are fully determined by the Rust router.
        batched = set(mm_proto.batched_keys)
        flat = dict(mm_proto.flat_keys)
        fields_config: dict[str, MultiModalFieldConfig] = {}
        flat_sizes_cache: dict[str, torch.Tensor] = {}
        for key in hf_dict:
            on_cpu = key in cpu_keys
            if key in batched:
                fields_config[key] = MultiModalFieldConfig.batched("image", keep_on_cpu=on_cpu)
            elif key in flat:
                sizes_key = flat[key]
                if sizes_key not in flat_sizes_cache:
                    flat_sizes_cache[sizes_key] = hf_dict[sizes_key].flatten().to(torch.int64)
                fields_config[key] = MultiModalFieldConfig.flat_from_sizes(
                    "image", flat_sizes_cache[sizes_key], keep_on_cpu=on_cpu
                )
            else:
                fields_config[key] = MultiModalFieldConfig.shared("image", num_images)

        batch_feature = BatchFeature(hf_dict, tensor_type="pt")
        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(batch_feature, fields_config)

        # Build mm_hashes: dict[str, list[str]]
        mm_hashes: dict[str, list[str]] = {}
        if mm_proto.mm_hashes:
            mm_hashes["image"] = list(mm_proto.mm_hashes)

        # Build mm_placeholders: dict[str, list[PlaceholderRange]]
        # When structural tokens (e.g. <|image_start|>, separators) are present
        # in the placeholder range, we must set is_embed so Aphrodite only scatters
        # encoder embeddings into patch-token positions (im_token_id).
        mm_placeholders: dict[str, list[PlaceholderRange]] = {}
        if mm_proto.mm_placeholders:
            im_token_id = mm_proto.im_token_id if mm_proto.HasField("im_token_id") else None
            # Pre-convert to tensor for vectorized mask building
            prompt_ids_tensor = torch.tensor(prompt_token_ids, dtype=torch.int64) if im_token_id is not None else None
            placeholders = []
            for p in mm_proto.mm_placeholders:
                is_embed = None
                if prompt_ids_tensor is not None:
                    mask = prompt_ids_tensor[p.offset : p.offset + p.length] == im_token_id
                    # Only set is_embed when there are non-embed positions,
                    # otherwise None means "all positions are embeds" which is
                    # both correct and avoids unnecessary overhead.
                    if not mask.all():
                        is_embed = mask
                placeholders.append(PlaceholderRange(offset=p.offset, length=p.length, is_embed=is_embed))
            mm_placeholders["image"] = placeholders

        return mm_input(
            prompt_token_ids=prompt_token_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
            prompt=tokenized.original_text or None,
        )

    @staticmethod
    def _sampling_params_from_proto(
        params: vllm_engine_pb2.SamplingParams,
        stream: bool = True,
        kv_transfer_params: vllm_engine_pb2.KvTransferParams | None = None,
    ) -> SamplingParams:
        """
        Convert protobuf SamplingParams to Aphrodite SamplingParams.

        Args:
            params: Protobuf SamplingParams message
            stream: Whether streaming is enabled
            kv_transfer_params: KV transfer params proto for Mooncake PD

        Returns:
            Aphrodite SamplingParams with detokenize=False and structured_outputs
        """
        # Build stop sequences
        stop = list(params.stop) if params.stop else None
        stop_token_ids = list(params.stop_token_ids) if params.stop_token_ids else None

        # Handle structured outputs constraints
        structured_outputs = None
        constraint_field = params.WhichOneof("constraint")
        if constraint_field:
            if constraint_field == "json_schema":
                structured_outputs = StructuredOutputsParams(json=params.json_schema)
            elif constraint_field == "regex":
                structured_outputs = StructuredOutputsParams(regex=params.regex)
            elif constraint_field == "grammar":
                structured_outputs = StructuredOutputsParams(grammar=params.grammar)
            elif constraint_field == "structural_tag":
                structured_outputs = StructuredOutputsParams(structural_tag=params.structural_tag)
            elif constraint_field == "json_object":
                structured_outputs = StructuredOutputsParams(json_object=params.json_object)
            elif constraint_field == "choice":
                structured_outputs = StructuredOutputsParams(choice=list(params.choice.choices))

        # Build extra_args for kv_transfer_params (Mooncake PD)
        extra_args = None
        if kv_transfer_params:
            remote_host = kv_transfer_params.remote_host
            remote_port = kv_transfer_params.remote_port
            if not remote_host or not (1 <= remote_port <= 65535):
                raise ValueError(
                    "Invalid kv_transfer_params: remote_host must be set and remote_port must be in [1, 65535]."
                )
            logger.debug(
                "kv_transfer_params={remote_host=%s, remote_port=%d}",
                remote_host,
                remote_port,
            )
            extra_args = {
                "kv_transfer_params": {
                    "remote_host": remote_host,
                    "remote_port": remote_port,
                }
            }

        # Create SamplingParams
        # output_kind=DELTA: Return only new tokens in each chunk (for streaming)
        return SamplingParams(
            temperature=params.temperature if params.HasField("temperature") else 1.0,
            top_p=params.top_p if params.top_p != 0.0 else 1.0,
            top_k=params.top_k,
            min_p=params.min_p,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty,
            repetition_penalty=params.repetition_penalty if params.repetition_penalty != 0.0 else 1.0,
            max_tokens=params.max_tokens if params.HasField("max_tokens") else None,
            min_tokens=params.min_tokens,
            stop=stop,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=params.skip_special_tokens,
            spaces_between_special_tokens=params.spaces_between_special_tokens,
            ignore_eos=params.ignore_eos,
            n=params.n if params.n > 0 else 1,
            logprobs=params.logprobs if params.HasField("logprobs") else None,
            prompt_logprobs=params.prompt_logprobs if params.HasField("prompt_logprobs") else None,
            seed=params.seed if params.HasField("seed") else None,
            include_stop_str_in_output=params.include_stop_str_in_output,
            logit_bias=dict(params.logit_bias) if params.logit_bias else None,
            structured_outputs=structured_outputs,
            extra_args=extra_args,
            # detokenize must be True if stop strings are used
            detokenize=bool(stop),
            output_kind=RequestOutputKind.DELTA if stream else RequestOutputKind.FINAL_ONLY,
        )

    @staticmethod
    def _build_top_logprobs(
        logprob_entry: dict[int, Logprob] | None,
        num_top_logprobs: int | None,
    ) -> vllm_engine_pb2.TopLogProbs:
        """Build TopLogProbs proto from a logprob entry dict."""
        top = vllm_engine_pb2.TopLogProbs()
        if num_top_logprobs and logprob_entry:
            limit = None if num_top_logprobs == -1 else num_top_logprobs
            for tid, lp in itertools.islice(logprob_entry.items(), limit):
                top.token_ids.append(tid)
                top.values.append(lp.logprob)
        return top

    @staticmethod
    def _build_output_logprobs(
        logprobs: SampleLogprobs | None,
        token_ids: Sequence[int],
        num_top_logprobs: int | None,
    ) -> vllm_engine_pb2.OutputLogProbs | None:
        """
        Convert Aphrodite SampleLogprobs to proto OutputLogProbs.

        Args:
            logprobs: Aphrodite logprobs (list of dict[int, Logprob])
            token_ids: Token IDs for each position
            num_top_logprobs: Number of top logprobs to include

        Returns:
            OutputLogProbs proto or None
        """
        if not logprobs:
            return None

        proto = vllm_engine_pb2.OutputLogProbs()

        for token_id, logprob_entry in zip(token_ids, logprobs):
            if logprob_entry is not None and (logprob := logprob_entry.get(token_id)):
                proto.token_logprobs.append(logprob.logprob)
                proto.token_ids.append(token_id)

                if num_top_logprobs:
                    proto.top_logprobs.append(
                        AphroditeEngineServicer._build_top_logprobs(logprob_entry, num_top_logprobs)
                    )

        return proto if proto.token_ids else None

    @staticmethod
    def _build_input_logprobs(
        prompt_logprobs: PromptLogprobs | None,
        prompt_token_ids: Sequence[int] | None,
        num_top_logprobs: int | None,
    ) -> vllm_engine_pb2.InputLogProbs | None:
        """
        Convert Aphrodite PromptLogprobs to proto InputLogProbs.

        Args:
            prompt_logprobs: Aphrodite prompt logprobs (list of dict[int, Logprob] | None)
            prompt_token_ids: Prompt token IDs
            num_top_logprobs: Number of top logprobs to include

        Returns:
            InputLogProbs proto or None
        """
        if not prompt_logprobs or prompt_token_ids is None:
            return None

        proto = vllm_engine_pb2.InputLogProbs()

        for token_id, logprob_entry in zip(prompt_token_ids, prompt_logprobs):
            token_logprob = vllm_engine_pb2.InputTokenLogProb()

            # First token has no logprob (None)
            if logprob_entry is not None and token_id in logprob_entry:
                token_logprob.value = logprob_entry[token_id].logprob

            proto.token_logprobs.append(token_logprob)
            proto.token_ids.append(token_id)
            if num_top_logprobs:
                proto.top_logprobs.append(AphroditeEngineServicer._build_top_logprobs(logprob_entry, num_top_logprobs))

        return proto if proto.token_ids else None

    @staticmethod
    def _tokenization_kwargs_from_proto(
        params: vllm_engine_pb2.SamplingParams,
    ) -> dict[str, int] | None:
        if params.HasField("truncate_prompt_tokens"):
            return {"truncate_prompt_tokens": params.truncate_prompt_tokens}
        return None

    @staticmethod
    def _chunk_response(
        output: RequestOutput,
        completion: "CompletionOutput | None" = None,
        num_logprobs: int | None = None,
        num_prompt_logprobs: int | None = None,
        is_first_chunk: bool = False,
    ) -> vllm_engine_pb2.GenerateResponse:
        """
        Build a streaming chunk response from Aphrodite output.
        When output_kind=DELTA, Aphrodite returns only new tokens automatically.

        Note: This sends DELTA logprobs (only for new tokens in this chunk).
        The Rust side is responsible for accumulating if needed.

        Args:
            output: Aphrodite RequestOutput (with delta tokens when output_kind=DELTA)
            completion: Specific CompletionOutput to use (for n>1 support).
                       If None, uses output.outputs[0] for backwards compatibility.
            num_logprobs: Number of top logprobs for output tokens
            num_prompt_logprobs: Number of top logprobs for prompt tokens
            is_first_chunk: Whether this is the first chunk for this index
                           (include input_logprobs only on first chunk)

        Returns:
            GenerateResponse with chunk field set
        """
        # Use provided completion or fall back to first output
        if completion is None:
            completion = output.outputs[0] if output.outputs else None

        if completion is None:
            # Empty chunk
            return vllm_engine_pb2.GenerateResponse(
                chunk=vllm_engine_pb2.GenerateStreamChunk(
                    token_ids=[],
                    prompt_tokens=0,
                    completion_tokens=0,
                    cached_tokens=0,
                    index=0,
                ),
            )

        # Build output logprobs for this chunk's tokens (delta, not cumulative)
        output_logprobs = AphroditeEngineServicer._build_output_logprobs(
            completion.logprobs, completion.token_ids, num_logprobs
        )

        # Build input logprobs only on first chunk for this index
        input_logprobs = None
        if is_first_chunk:
            input_logprobs = AphroditeEngineServicer._build_input_logprobs(
                output.prompt_logprobs,
                output.prompt_token_ids,
                num_prompt_logprobs,
            )

        # When output_kind=DELTA, completion.token_ids contains only new tokens
        # Aphrodite handles the delta logic internally
        # completion_tokens = delta count (client will accumulate)
        return vllm_engine_pb2.GenerateResponse(
            chunk=vllm_engine_pb2.GenerateStreamChunk(
                token_ids=completion.token_ids,
                prompt_tokens=len(output.prompt_token_ids) if output.prompt_token_ids else 0,
                completion_tokens=len(completion.token_ids),  # Delta count
                cached_tokens=output.num_cached_tokens,
                output_logprobs=output_logprobs,
                input_logprobs=input_logprobs,
                index=completion.index,
            ),
        )

    @staticmethod
    def _complete_response(
        output: RequestOutput,
        completion: "CompletionOutput | None" = None,
        num_logprobs: int | None = None,
        num_prompt_logprobs: int | None = None,
    ) -> vllm_engine_pb2.GenerateResponse:
        """
        Build a final completion response from Aphrodite output.

        For non-streaming (FINAL_ONLY): completion has all tokens and logprobs.
        For streaming (DELTA): completion has last delta; Rust accumulates.

        Args:
            output: Aphrodite RequestOutput (finished=True)
            completion: Specific CompletionOutput to use (for n>1 support).
                       If None, uses output.outputs[0] for backwards compatibility.
            num_logprobs: Number of top logprobs for output tokens
            num_prompt_logprobs: Number of top logprobs for prompt tokens

        Returns:
            GenerateResponse with complete field set
        """
        # Use provided completion or fall back to first output
        if completion is None:
            completion = output.outputs[0] if output.outputs else None

        if completion is None:
            # Empty completion
            return vllm_engine_pb2.GenerateResponse(
                complete=vllm_engine_pb2.GenerateComplete(
                    output_ids=[],
                    finish_reason="error",
                    prompt_tokens=0,
                    completion_tokens=0,
                    cached_tokens=0,
                    index=0,
                ),
            )

        # Build output logprobs from completion's data
        # For non-streaming: this has all logprobs
        # For streaming: this has only last delta (Rust accumulates from chunks)
        output_logprobs = AphroditeEngineServicer._build_output_logprobs(
            completion.logprobs, completion.token_ids, num_logprobs
        )

        # Build input logprobs
        input_logprobs = AphroditeEngineServicer._build_input_logprobs(
            output.prompt_logprobs,
            output.prompt_token_ids,
            num_prompt_logprobs,
        )

        # Build kv_transfer_params if present (Mooncake PD)
        kv_transfer_params = None
        if output.kv_transfer_params:
            kv_transfer_params = vllm_engine_pb2.KvTransferParams(
                remote_host=output.kv_transfer_params.get("remote_host", ""),
                remote_port=output.kv_transfer_params.get("remote_port", 0),
            )

        # Build matched_stop kwargs from stop_reason (int token ID or str stop sequence)
        stop_kwargs: dict[str, int | str] = {}
        if completion.stop_reason is not None:
            if isinstance(completion.stop_reason, int):
                stop_kwargs["matched_token_id"] = completion.stop_reason
            else:
                stop_kwargs["matched_stop_str"] = str(completion.stop_reason)

        # Build complete response
        # When streaming (DELTA mode): completion.token_ids will be empty/last delta
        # When non-streaming (FINAL_ONLY mode): completion.token_ids has all tokens
        # Client will accumulate token counts for streaming
        return vllm_engine_pb2.GenerateResponse(
            complete=vllm_engine_pb2.GenerateComplete(
                output_ids=completion.token_ids,
                finish_reason=completion.finish_reason or "stop",
                prompt_tokens=len(output.prompt_token_ids) if output.prompt_token_ids else 0,
                completion_tokens=len(completion.token_ids),
                cached_tokens=output.num_cached_tokens,
                output_logprobs=output_logprobs,
                input_logprobs=input_logprobs,
                index=completion.index,
                kv_transfer_params=kv_transfer_params,
                **stop_kwargs,
            ),
        )
