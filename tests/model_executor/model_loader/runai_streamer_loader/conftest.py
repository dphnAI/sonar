# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from aphrodite.utils.network_utils import get_distributed_init_method, get_ip, get_open_port
from aphrodite.v1.executor import UniProcExecutor
from aphrodite.v1.worker.worker_base import WorkerWrapperBase


# This is a dummy executor for patching in test_runai_model_streamer_s3.py.
# We cannot use aphrodite_runner fixture here, because it spawns worker process.
# The worker process reimports the patched entities, and the patch is not applied.
class RunaiDummyExecutor(UniProcExecutor):
    def _init_executor(self) -> None:
        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())

        local_rank = 0
        rank = 0
        is_driver_worker = True

        device_info = self.aphrodite_config.device_config.device.__str__().split(":")
        if len(device_info) > 1:
            local_rank = int(device_info[1])

        worker_rpc_kwargs = dict(
            aphrodite_config=self.aphrodite_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        self.driver_worker = WorkerWrapperBase()

        self.collective_rpc("init_worker", args=([worker_rpc_kwargs],))
        self.collective_rpc("init_device")
