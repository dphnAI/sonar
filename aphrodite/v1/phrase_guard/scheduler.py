# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, fields, replace

from aphrodite.logger import init_logger
from aphrodite.sampling_params import SamplingParams
from aphrodite.tokenizers.registry import cached_tokenizer_from_config
from aphrodite.v1.core.sched.async_scheduler import AsyncScheduler
from aphrodite.v1.core.sched.output import SchedulerOutput
from aphrodite.v1.core.sched.utils import check_stop
from aphrodite.v1.engine.detokenizer import FastIncrementalDetokenizer, IncrementalDetokenizer
from aphrodite.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)
from aphrodite.v1.phrase_guard.logprobs import PendingLogprobs, join_logprobs
from aphrodite.v1.phrase_guard.matcher import MAX_RETRIES, PendingText, validate_phrases
from aphrodite.v1.phrase_guard.processor import RETRY_KEY, PhraseRetryProcessor
from aphrodite.v1.request import RequestStatus

logger = init_logger(__name__)


@dataclass
class PhraseSchedulerOutput(SchedulerOutput):
    phrase_rewinds: dict[str, tuple[list[int], SamplingParams]] | None = None


class RequestGuard:
    def __init__(self, request, tokenizer):
        self.request = request
        self.tokenizer = tokenizer
        args = request.sampling_params.extra_args
        self.pending = PendingText(
            validate_phrases(args["banned_strings"]), bool(args.get("banned_strings_case_sensitive", False))
        )
        self.retries = 0
        self.retry_position = -1
        self.blocked: list[int] = []
        self.logprobs = PendingLogprobs() if request.sampling_params.num_logprobs is not None else None
        self.step_logprobs = None
        self.output_logprobs = None
        self.prompt_logprobs = None
        self.reset_decoder()

    def reset_decoder(self):
        self.decoder = IncrementalDetokenizer.from_new_request(self.tokenizer, self.request)
        if not isinstance(self.decoder, FastIncrementalDetokenizer):
            raise ValueError("Experimental banned_strings requires a fast Hugging Face tokenizer")
        for token in self.request.output_token_ids:
            self.decode(token)

    def decode(self, token):
        self.decoder.token_ids.append(token)
        return self.decoder.decode_next(token)


class PhraseScheduler(AsyncScheduler):
    """Async phrase rollback with fenced writes and immutable cached prefixes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.aphrodite_config
        if not config.scheduler_config.async_scheduling:
            raise ValueError("PhraseScheduler requires async_scheduling=True")
        if self.connector is not None or self.ec_connector is not None:
            raise ValueError("Experimental PhraseScheduler does not support cache connectors")
        if not self.cache_config.enable_prefix_caching:
            raise ValueError("PhraseScheduler requires prefix caching for replay")
        if any(
            type(group.kv_cache_spec)
            not in (
                FullAttentionSpec,
                MLAAttentionSpec,
                MambaSpec,
                SlidingWindowSpec,
                SlidingWindowMLASpec,
                ChunkedLocalAttentionSpec,
            )
            for group in self.kv_cache_config.kv_cache_groups
        ):
            raise ValueError("Unsupported cache layout for experimental PhraseScheduler")
        self.guards: dict[str, RequestGuard] = {}
        self.rewinds: dict[str, tuple[list[int], SamplingParams]] = {}
        self.tokenizer = None
        self.defer_block_free = True
        self.logprob_guards: set[str] = set()

    def add_request(self, request):
        params = request.sampling_params
        if params and "banned_strings" in (params.extra_args or {}):
            PhraseRetryProcessor.validate_params(params)
            processors = self.aphrodite_config.model_config.logits_processors or []
            if not self.use_v2_model_runner and not any(
                p is PhraseRetryProcessor or str(p).endswith("PhraseRetryProcessor") for p in processors
            ):
                raise ValueError("banned_strings requires the PhraseRetryProcessor logits processor")
            if (
                self.return_sampling_mask
                or self.enable_return_routed_experts
                or request.resumable
                or request.has_encoder_inputs
                or request.prompt_embeds is not None
            ):
                raise ValueError("Experimental banned_strings requires non-resumable plain text generation")
            if self.tokenizer is None:
                self.tokenizer = cached_tokenizer_from_config(self.aphrodite_config.model_config)
            self.guards[request.request_id] = RequestGuard(request, self.tokenizer)
            if params.num_logprobs is not None or params.prompt_logprobs is not None:
                self.logprob_guards.add(request.request_id)
        super().add_request(request)

    def schedule(self, throttle_prefills=False):
        # Rewound histories must not reach workers until old frames drain.
        held = []
        if self.rewinds:
            held = [r for r in self.waiting if r.request_id in self.rewinds and r.num_in_flight_tokens]
            self.waiting.remove_requests(held)
            for rid in self.rewinds:
                # Retry the checkpoint without drafts from the discarded history.
                self.requests[rid].spec_token_ids = []
        try:
            output = super().schedule(throttle_prefills)
        finally:
            for request in reversed(held):
                self.waiting.prepend_request(request)
        rewinds = (
            {rid: self.rewinds.pop(rid) for rid in output.num_scheduled_tokens if rid in self.rewinds}
            if self.rewinds
            else None
        )
        if rewinds:
            if self.use_v2_model_runner:
                for request in output.scheduled_new_reqs:
                    if request.req_id in rewinds:
                        request.sampling_params = rewinds[request.req_id][1]
            else:
                output = PhraseSchedulerOutput(
                    **{f.name: getattr(output, f.name) for f in fields(SchedulerOutput)}, phrase_rewinds=rewinds
                )
        return output

    def _update_request_with_output(self, request, new_token_ids, is_stale=False):
        guard = self.guards.get(request.request_id)
        if guard is None:
            return super()._update_request_with_output(request, new_token_ids, is_stale)
        if not is_stale:
            request.num_output_placeholders -= len(new_token_ids)
            assert request.num_output_placeholders >= 0
        ready = []
        ready_logprobs = []
        stopped = False
        rewound = False
        for index, token in enumerate(new_token_ids):
            if request.num_output_tokens == guard.retry_position and token in guard.blocked:
                logger.error("Phrase retry exhausted sampling constraints for request %s", request.request_id)
                request.status = RequestStatus.FINISHED_ERROR
                stopped = True
                break
            request.append_output_token_ids(token)
            if guard.logprobs is not None:
                assert guard.step_logprobs is not None
                guard.logprobs.append(guard.step_logprobs, index)
            params = request.sampling_params
            stop_token = token == params.eos_token_id or token in (params.stop_token_ids or ())
            if stop_token and not params.include_stop_str_in_output:
                # The output detokenizer omits this token's text. Flush the
                # pending prefix and let it consume the stop token normally.
                final = guard.pending.finish() + [token]
                ready.extend(final)
                if guard.logprobs is not None:
                    ready_logprobs.extend(guard.logprobs.take(len(final)))
                check_stop(request, self.max_model_len)
                stopped = True
                break
            try:
                decision = guard.pending.feed(token, guard.decode(token))
            except ValueError:
                logger.exception("Phrase guard exhausted its buffer for request %s", request.request_id)
                request.status = RequestStatus.FINISHED_ERROR
                stopped = True
                break
            ready.extend(decision.ready)
            if guard.logprobs is not None:
                ready_logprobs.extend(guard.logprobs.take(len(decision.ready)))
            if decision.rewind_to is not None:
                if guard.logprobs is not None:
                    guard.logprobs.clear()
                if stop_token:
                    logger.error(
                        "Banned phrase conflicts with an included stop token for request %s", request.request_id
                    )
                    request.status = RequestStatus.FINISHED_ERROR
                    stopped = True
                    break
                guard.retries += 1
                if guard.retries > MAX_RETRIES:
                    request.status = RequestStatus.FINISHED_ERROR
                    stopped = True
                    break
                self._rewind(request, guard, decision)
                rewound = True
                break
            if check_stop(request, self.max_model_len):
                final = guard.pending.finish()
                ready.extend(final)
                if guard.logprobs is not None:
                    ready_logprobs.extend(guard.logprobs.take(len(final)))
                stopped = True
                break
        guard.output_logprobs = join_logprobs(ready_logprobs)
        if not stopped and not rewound and request.status == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens - request.num_output_placeholders)
        return ready, stopped

    def _rewind(self, request, guard, decision):
        position = decision.rewind_to
        if guard.retry_position != position:
            guard.retry_position = position
            guard.blocked.clear()
        guard.blocked.append(decision.blocked_token)
        # Resume through cache lookup; published attention blocks and recurrent
        # checkpoints stay immutable until old GPU writes have drained.
        if request.status == RequestStatus.RUNNING:
            self.running.remove(request)
            self._preempt_request(request, 0.0, drop_stale_output=True)
        else:
            request.drop_stale_output = True
        del request._output_token_ids[position:]
        del request._all_token_ids[request.num_prompt_tokens + position :]
        del request.block_hashes[request.num_tokens // self.hash_block_size :]
        params = request.sampling_params.clone()
        if self.use_v2_model_runner:
            # Original prompt probabilities are already buffered or published.
            params.prompt_logprobs = None
        params.extra_args = dict(params.extra_args or {})
        params.extra_args[RETRY_KEY] = (position, guard.blocked.copy())
        if self.use_v2_model_runner:
            # Preserve the retry if memory pressure preempts replay again.
            request.sampling_params = params
        self.rewinds[request.request_id] = (list(request.output_token_ids), params)
        guard.reset_decoder()
        logger.debug("Phrase rewind: request=%s position=%d retries=%d", request.request_id, position, guard.retries)

    def update_from_output(self, scheduler_output, model_runner_output):
        if not self.logprob_guards:
            return super().update_from_output(scheduler_output, model_runner_output)
        guards = {rid: self.guards[rid] for rid in model_runner_output.req_ids if rid in self.logprob_guards}
        prompt_logprobs = dict(model_runner_output.prompt_logprobs_dict)
        for rid, guard in guards.items():
            guard.step_logprobs = None
            guard.output_logprobs = None
            if guard.logprobs is not None and model_runner_output.logprobs is not None:
                index = model_runner_output.req_id_to_index[rid]
                guard.step_logprobs = model_runner_output.logprobs.slice_request(
                    index, len(model_runner_output.sampled_token_ids[index])
                )
            if rid in prompt_logprobs:
                guard.prompt_logprobs = prompt_logprobs.pop(rid)
        outputs = super().update_from_output(
            scheduler_output, replace(model_runner_output, prompt_logprobs_dict=prompt_logprobs)
        )
        for batch in outputs.values():
            for output in batch.outputs:
                if guard := guards.get(output.request_id):
                    output.new_logprobs = guard.output_logprobs
                    output.new_prompt_logprobs_tensors = guard.prompt_logprobs
                    guard.prompt_logprobs = None
        for guard in guards.values():
            guard.step_logprobs = None
        return outputs

    def _free_request(self, request, delay_free_blocks=False):
        result = super()._free_request(request, delay_free_blocks)
        self.guards.pop(request.request_id, None)
        self.rewinds.pop(request.request_id, None)
        self.logprob_guards.discard(request.request_id)
        return result
