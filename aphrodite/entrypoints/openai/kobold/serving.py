# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import AsyncGenerator

from fastapi import Request

from aphrodite.engine.protocol import EngineClient
from aphrodite.entrypoints.serve.utils.request_logger import RequestLogger
from aphrodite.entrypoints.openai.engine.serving import OpenAIServing
from aphrodite.entrypoints.openai.kobold.protocol import KAIGenerationInputSchema
from aphrodite.entrypoints.openai.models.serving import OpenAIServingModels
from aphrodite.logger import init_logger
from aphrodite.sampling_params import SamplingParams
from aphrodite.tokenizers import TokenizerLike
from aphrodite.utils import random_uuid

logger = init_logger(__name__)

_SAMPLING_EPS = 1e-5
_KOBOLD_BADWORD_BIAS = -100.0
gen_cache: dict[str, str] = {}


class OpenAIServingKobold(OpenAIServing):
    """Serving class for KoboldAI API compatibility."""

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None = None,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
        )
        self._initialize_badwordsids()

    def _initialize_badwordsids(self) -> None:
        self.badwordsids: list[int] = []

        hf_config = getattr(self.model_config, "hf_config", None)
        if hf_config and getattr(hf_config, "bad_words_ids", None):
            raw_bad_words = hf_config.bad_words_ids
            flattened: list[int] = []
            for item in raw_bad_words:
                if isinstance(item, int):
                    flattened.append(item)
                elif isinstance(item, list):
                    flattened.extend(token_id for token_id in item if isinstance(token_id, int))
            self.badwordsids = flattened
            return

        try:
            tokenizer = self.renderer.get_tokenizer()
            vocab = tokenizer.get_vocab()
            bracket_tokens = [token_id for token, token_id in vocab.items() if any(ch in str(token) for ch in "[]")]
            self.badwordsids = bracket_tokens

            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is not None and pad_token_id in self.badwordsids:
                self.badwordsids.remove(pad_token_id)

            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if eos_token_id is not None and eos_token_id not in self.badwordsids:
                self.badwordsids.append(eos_token_id)
        except Exception as e:
            logger.warning("Could not initialize badwordsids from tokenizer: %s", e)
            self.badwordsids = []

    async def create_kobold_response(
        self,
        request: KAIGenerationInputSchema,
        raw_request: Request | None = None,
    ) -> dict[str, list[dict[str, str]]]:
        tokenizer = self.renderer.get_tokenizer()
        sampling_params, input_tokens = self._prepare_engine_payload(request, tokenizer)
        request_id = request.genkey or f"kai-{random_uuid()}"

        results_generator = self.engine_client.generate(
            {
                "prompt": request.prompt,
                "prompt_token_ids": input_tokens,
            },
            sampling_params,
            request_id,
        )

        final_res = None
        previous_output = ""
        async for res in results_generator:
            final_res = res
            new_chunk = res.outputs[0].text[len(previous_output) :]
            previous_output += new_chunk
            if request.genkey:
                gen_cache[request.genkey] = previous_output

        assert final_res is not None
        if request.genkey:
            gen_cache.pop(request.genkey, None)

        return {"results": [{"text": output.text} for output in final_res.outputs]}

    async def create_kobold_stream(
        self,
        request: KAIGenerationInputSchema,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None]:
        tokenizer = self.renderer.get_tokenizer()
        sampling_params, input_tokens = self._prepare_engine_payload(request, tokenizer)
        request_id = request.genkey or f"kai-{random_uuid()}"

        results_generator = self.engine_client.generate(
            {
                "prompt": request.prompt,
                "prompt_token_ids": input_tokens,
            },
            sampling_params,
            request_id,
        )

        previous_output = ""
        async for res in results_generator:
            new_chunk = res.outputs[0].text[len(previous_output) :]
            previous_output += new_chunk
            if request.genkey:
                gen_cache[request.genkey] = previous_output
            yield f"event: message\ndata: {json.dumps({'token': new_chunk})}\n\n"

    async def check_generation(self, genkey: str) -> str:
        return gen_cache.get(genkey, "")

    async def abort_generation(self, genkey: str) -> None:
        gen_cache.pop(genkey, None)
        await self.engine_client.abort(genkey)

    def _prepare_engine_payload(
        self,
        kai_payload: KAIGenerationInputSchema,
        tokenizer: TokenizerLike,
    ) -> tuple[SamplingParams, list[int]]:
        if not kai_payload.genkey:
            kai_payload.genkey = f"kai-{random_uuid()}"

        top_k = kai_payload.top_k if (kai_payload.top_k or 0) != 0 else -1
        tfs = max(_SAMPLING_EPS, kai_payload.tfs or 0.0)
        top_p = kai_payload.top_p or 1.0
        n = kai_payload.n or 1

        if (kai_payload.temperature or 0.0) < _SAMPLING_EPS:
            n = 1
            top_p = 1.0
            top_k = -1

        logit_bias = None
        if getattr(kai_payload, "use_default_badwordsids", True) and self.badwordsids:
            logit_bias = {token_id: _KOBOLD_BADWORD_BIAS for token_id in self.badwordsids}

        sampling_params = SamplingParams(
            n=n,
            repetition_penalty=kai_payload.rep_pen or 1.0,
            temperature=kai_payload.temperature or 1.0,
            smoothing_factor=kai_payload.smoothing_factor or 0.0,
            smoothing_curve=kai_payload.smoothing_curve or 1.0,
            tfs=tfs,
            top_p=top_p,
            top_k=top_k,
            top_a=kai_payload.top_a or 0.0,
            min_p=kai_payload.min_p or 0.0,
            typical_p=kai_payload.typical or 1.0,
            eta_cutoff=kai_payload.eta_cutoff or 0.0,
            epsilon_cutoff=kai_payload.eps_cutoff or 0.0,
            stop=kai_payload.stop_sequence,
            include_stop_str_in_output=kai_payload.include_stop_str_in_output or False,
            logit_bias=logit_bias,
            max_tokens=kai_payload.max_length,
            seed=kai_payload.sampler_seed,
            xtc_probability=kai_payload.xtc_probability or 0.0,
            xtc_threshold=kai_payload.xtc_threshold or 0.0,
            skip_clone=True,
        )

        max_input_tokens = max(1, kai_payload.max_context_length - kai_payload.max_length)
        input_tokens = tokenizer.encode(
            kai_payload.prompt,
            add_special_tokens=False,
        )[-max_input_tokens:]
        return sampling_params, input_tokens
