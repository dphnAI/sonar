# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run harness evaluations against Sonar, without importing the vLLM runtime."""

import json
from dataclasses import replace
from typing import Any

import lm_eval
from lm_eval.models.openai_completions import LocalCompletionsAPI
from lm_eval.models.utils import _add_special_kwargs, has_bos_prefix, postprocess_generated_text

from tests.utils import RemoteOpenAIServer


class _SonarCompletionsAPI(LocalCompletionsAPI):
    def __init__(self, *, chat_template_args=None, enable_thinking=None, think_end_token=None, **kwargs):
        self.chat_template_args = dict(chat_template_args or {})
        if enable_thinking is not None:
            self.chat_template_args["enable_thinking"] = enable_thinking
        self.think_end_token = think_end_token
        super().__init__(**kwargs)

    def tok_encode(self, string, left_truncate_len=None, add_special_tokens=None, truncation=False, **kwargs):
        if not string:
            return []
        strings = [string] if isinstance(string, str) else string
        prefix_id = self.tokenizer.bos_token_id
        if prefix_id is None:
            prefix_id = self.tokenizer.eos_token_id
        prefix = self.tokenizer.decode(prefix_id) if prefix_id is not None else None
        special_kwargs = _add_special_kwargs(add_special_tokens, self.add_bos_token)
        encodings = []
        for text in strings:
            token_kwargs = dict(special_kwargs)
            if has_bos_prefix(text, prefix):
                token_kwargs["add_special_tokens"] = False
            encoding = self.tokenizer(
                text,
                return_attention_mask=False,
                truncation=truncation,
                **token_kwargs,
                **kwargs,
            ).input_ids
            encodings.append(encoding[-left_truncate_len:] if left_truncate_len else encoding)
        return encodings[0] if isinstance(string, str) else encodings

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
            **self.chat_template_args,
        )

    def generate_until(self, requests, disable_tqdm=False):
        if not self.think_end_token:
            return super().generate_until(requests, disable_tqdm=disable_tqdm)

        # Task stop strings can appear in reasoning. Apply them only after the
        # thinking section, as the offline harness adapter does.
        generation_requests = [
            replace(request, arguments=(request.args[0], {**request.args[1], "until": [self.eos_string]}))
            for request in requests
        ]
        outputs = super().generate_until(generation_requests, disable_tqdm=disable_tqdm)
        return [
            postprocess_generated_text(output, request.args[1].get("until"), self.think_end_token)
            for output, request in zip(outputs, requests, strict=True)
        ]


def evaluate_sonar(model_args: dict[str, Any], **evaluation_kwargs):
    """Evaluate a real Sonar server using the harness's supported completions API.

    `model_args` uses the existing offline evaluation convention: `pretrained`
    identifies the model, engine options configure Sonar, and tokenizer/chat
    options configure the harness rather than being forwarded to the engine.
    """
    engine_args = dict(model_args)
    engine_args.setdefault("seed", 1234)
    model_name = engine_args.pop("pretrained")
    harness_args = {"add_bos_token": None}
    for name in ("add_bos_token", "chat_template_args", "enable_thinking", "think_end_token"):
        if name in engine_args:
            harness_args[name] = engine_args.pop(name)
    tokenizer_args = {
        "tokenizer": engine_args.get("tokenizer"),
        "revision": engine_args.get("tokenizer_revision") or engine_args.get("revision", "main"),
        "seed": engine_args["seed"],
    }
    trust_remote_code = engine_args.get("trust_remote_code", False)
    server_args = []
    for name, value in engine_args.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            server_args.append(flag if value else "--no-" + name.replace("_", "-"))
        elif isinstance(value, list):
            server_args.extend([flag, *(str(item) for item in value)])
        elif value is not None:
            server_args.extend([flag, json.dumps(value) if isinstance(value, dict) else str(value)])

    batch_size = evaluation_kwargs.get("batch_size", 1)
    num_concurrent = 64 if batch_size == "auto" else int(batch_size)
    with RemoteOpenAIServer(model_name, server_args, max_wait_seconds=1200) as server:
        with server.get_client() as client:
            model_card = client.models.list().data[0]
            max_length = model_card.model_dump()["max_model_len"]
        model = _SonarCompletionsAPI(
            model=model_name,
            base_url=server.url_for("v1/completions"),
            tokenizer_backend="huggingface",
            tokenized_requests=True,
            trust_remote_code=trust_remote_code,
            max_length=max_length,
            batch_size=1,
            num_concurrent=num_concurrent,
            **tokenizer_args,
            **harness_args,
        )
        return lm_eval.simple_evaluate(model=model, **evaluation_kwargs)
