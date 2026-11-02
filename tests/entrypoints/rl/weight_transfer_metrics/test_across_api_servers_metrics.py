# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-transfer metrics scraped from a real two-API-server deployment."""

import requests
import torch
from prometheus_client.parser import text_string_to_metric_families
from transformers import AutoModelForCausalLM

from aphrodite.distributed.weight_transfer import (
    HTTPAphroditeWeightSyncClient,
    ModuleSource,
    WeightTransferTrainerFactory,
)
from aphrodite.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo
from tests.entrypoints.rl.conftest import MODEL_NAME
from tests.utils import RemoteOpenAIServer

DURATION = "aphrodite:rl_weight_update_operation_duration_seconds"
IN_FLIGHT = "aphrodite:rl_weight_update_operations_in_flight"
SYNCS = 8


def scrape(url: str) -> dict[tuple[str, str], float]:
    text = requests.get(f"{url}/metrics", timeout=30).text
    return {
        (sample.name, sample.labels["operation"]): sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name in (f"{DURATION}_count", IN_FLIGHT)
    }


def test_weight_sync_metrics_aggregate_across_api_servers(tmp_path):
    args = [
        "--load-format",
        "dummy",
        "--enforce-eager",
        "--max-model-len",
        "512",
        "--gpu-memory-utilization",
        "0.5",
        "--api-server-count",
        "2",
        "--weight-transfer-config",
        '{"backend": "ipc"}',
    ]
    env = {
        "APHRODITE_SERVER_DEV_MODE": "1",
        "APHRODITE_ALLOW_INSECURE_SERIALIZATION": "1",
        "PROMETHEUS_MULTIPROC_DIR": str(tmp_path),
    }
    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=env) as server:
        url = server.url_root
        trainer = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).cuda()
        engine = WeightTransferTrainerFactory.trainer_init(
            IPCTrainerInitInfo(rank=0),
            client=HTTPAphroditeWeightSyncClient(url),
            source=ModuleSource(trainer),
        )
        for _ in range(SYNCS):
            engine.send_weights()
        metrics = scrape(url)

    calls = {"init": 1, "start": SYNCS, "update": SYNCS, "finish": SYNCS}
    for operation, count in calls.items():
        assert metrics[(f"{DURATION}_count", operation)] == count, operation
        assert metrics[(IN_FLIGHT, operation)] == 0, operation
