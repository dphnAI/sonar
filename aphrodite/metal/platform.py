# SPDX-License-Identifier: Apache-2.0
"""Metal Platform implementation for Aphrodite."""

import logging
import platform as py_platform
from typing import TYPE_CHECKING

import psutil
import torch
from aphrodite.platforms.interface import DeviceCapability, Platform, PlatformEnum

from aphrodite.metal.config import (
    enable_contiguous_kv_fast_path,
    get_config,
    should_use_contiguous_kv_fast_path,
)

if TYPE_CHECKING:
    from aphrodite.config import AphroditeConfig
    from aphrodite.v1.attention.backend import AttentionBackend
    from aphrodite.v1.attention.backends.registry import AttentionBackendEnum
    from aphrodite.v1.attention.selector import AttentionSelectorConfig

logger = logging.getLogger(__name__)


class MetalPlatform(Platform):
    """Platform implementation for Apple Silicon Metal/MLX.

    This class provides Aphrodite with information about the Metal platform
    capabilities and handles device management.
    """

    _enum: PlatformEnum = PlatformEnum.METAL
    device_name: str = "metal"
    device_type: str = "metal"
    dispatch_key: str = "CPU"  # PyTorch dispatch key

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get the name of the Metal device.

        Args:
            device_id: Device index (ignored for Metal, single GPU)

        Returns:
            Device name string
        """
        try:
            import mlx.core as mx

            device = mx.default_device()
            return f"Apple Silicon ({device})"
        except ImportError:
            return "Apple Silicon (MLX not available)"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Get total memory available for the device.

        On Apple Silicon, this returns the fraction of unified memory
        configured for use by the plugin.

        Args:
            device_id: Device index (ignored for Metal)

        Returns:
            Total memory in bytes
        """
        config = get_config()
        total_memory = psutil.virtual_memory().total
        # In auto mode, report full memory - actual allocation is dynamic
        if config.is_auto_memory:
            return total_memory
        return int(total_memory * config.memory_fraction)

    @classmethod
    def get_device_available_memory(cls, device_id: int = 0) -> int:
        """Get available memory for the device.

        Args:
            device_id: Device index (ignored for Metal)

        Returns:
            Available memory in bytes
        """
        config = get_config()
        available = psutil.virtual_memory().available
        # In auto mode, report full available memory - actual allocation is dynamic
        if config.is_auto_memory:
            return available
        return int(available * config.memory_fraction)

    @classmethod
    def is_available(cls) -> bool:
        """Check if Metal platform is available.

        Returns:
            True if running on Apple Silicon with MLX support
        """
        # Check architecture
        if py_platform.machine() != "arm64":
            return False

        # Check OS
        if py_platform.system() != "Darwin":
            return False

        # Check MLX availability without mutating global device state
        try:
            import mlx.core as mx

            return bool(mx.metal.is_available())
        except (ImportError, AttributeError, RuntimeError):
            return False

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        """Get device compute capability.

        Returns a fake capability for compatibility with CUDA-centric code.

        Args:
            device_id: Device index (ignored)

        Returns:
            DeviceCapability with (major, minor) version
        """
        # Return a reasonable value for compatibility
        return DeviceCapability(major=8, minor=0)

    @classmethod
    def get_device_count(cls) -> int:
        """Get number of available devices.

        Apple Silicon has unified memory, so we expose a single device.

        Returns:
            Always 1 for Metal
        """
        return 1

    @classmethod
    def set_device(cls, device: torch.device | int) -> None:
        """Set the current device.

        Args:
            device: Device or index (must resolve to 0 for Metal)
        """
        device_id = device.index if isinstance(device, torch.device) else device
        device_id = 0 if device_id is None else device_id
        if device_id != 0:
            msg = f"Metal only supports device 0, got {device_id}"
            raise ValueError(msg)

        config = get_config()
        if config.use_mlx:
            import mlx.core as mx

            device_type = (
                mx.DeviceType.gpu if config.mlx_device == "gpu" else mx.DeviceType.cpu
            )
            mx.set_default_device(mx.Device(device_type))

    @classmethod
    def current_device(cls) -> int:
        """Get the current device index.

        Returns:
            Always 0 for Metal
        """
        return 0

    @classmethod
    def synchronize(cls, device_id: int = 0) -> None:
        """Synchronize the device.

        Args:
            device_id: Device index (ignored)
        """
        import mlx.core as mx

        # Prefer an explicit MLX barrier when available; otherwise force evaluation.
        # `mx.eval([])` is a no-op, so we evaluate a tiny scalar as a safe fallback.
        try:
            mx.synchronize()
        except (AttributeError, TypeError):
            mx.eval(mx.array(0, dtype=mx.int32))

        if torch.backends.mps.is_available():
            torch.mps.synchronize()

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        """Seed the Metal-side RNG (MLX) for this platform.

        Called from ``aphrodite.utils.torch_utils.set_random_seed`` after Python
        ``random``, NumPy, and PyTorch (which reaches MPS via its default
        generator) have all been seeded.  MLX maintains its own global PRNG
        that does not auto-seed and is not reached by ``torch.manual_seed``,
        so we seed it explicitly here.
        """
        import mlx.core as mx

        mx.random.seed(seed)

    @classmethod
    def get_torch_device(cls, device_id: int = 0) -> torch.device:
        """Get the corresponding PyTorch device.

        Args:
            device_id: Device index (ignored)

        Returns:
            PyTorch device (MPS or CPU)
        """
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @classmethod
    def check_and_update_config(cls, aphrodite_config: "AphroditeConfig") -> None:
        """Check and update Aphrodite configuration for Metal compatibility.

        Args:
            aphrodite_config: Aphrodite configuration object
        """
        config = get_config()
        parallel_config = aphrodite_config.parallel_config
        model_config = aphrodite_config.model_config
        compilation_config = aphrodite_config.compilation_config

        # Metal execution is MLX-backed. Torch Inductor/CUDAGraph settings do
        # not apply to the actual model path, so disable those compilation
        # surfaces without overriding the user's eager-mode flag here.
        from aphrodite.config.compilation import CompilationMode, CUDAGraphMode

        compilation_config.mode = CompilationMode.NONE
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        compilation_config.max_cudagraph_capture_size = 0
        compilation_config.cudagraph_capture_sizes = []
        compilation_config.cudagraph_num_of_warmups = 1

        # Apply TurboQuant config from --additional-config
        # Example: --additional-config '{"turboquant": true, "k_quant": "q4_0"}'
        add = getattr(aphrodite_config, "additional_config", None) or {}
        if add.get("turboquant"):
            config.turboquant = True
            config.k_quant = add.get("k_quant", "q8_0")
            config.v_quant = add.get("v_quant", "q3_0")
            config._validate_turboquant()
            logger.info(
                f"TurboQuant enabled via --additional-config: "
                f"k_quant={config.k_quant}, v_quant={config.v_quant}"
            )

        scheduler_config = aphrodite_config.scheduler_config
        if should_use_contiguous_kv_fast_path(
            config,
            model_config=model_config,
            scheduler_config=scheduler_config,
        ):
            enable_contiguous_kv_fast_path(config)
            logger.info(
                "Metal: using contiguous MLX KV cache for low-concurrency "
                "dense serving (max_num_seqs=%d). Set "
                "APHRODITE_METAL_USE_PAGED_ATTENTION=1 to force paged attention.",
                scheduler_config.max_num_seqs,
            )

        if config.debug:
            logger.info(f"Metal config: {config}")

        # Set worker class for Metal
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "aphrodite.v1.worker.metal_worker.MetalWorker"

        # Set executor backend (use uniproc for single device)
        if parallel_config.distributed_executor_backend in ("auto", None):
            parallel_config.distributed_executor_backend = "uni"

        # Disable features not supported on Metal
        parallel_config.disable_custom_all_reduce = True

        if getattr(scheduler_config, "enable_chunked_prefill", False):
            if config.use_paged_attention:
                # The paged path uses a unified varlen Metal kernel that
                # handles mixed prefill + decode in a single forward pass,
                # so chunked prefill works correctly.
                logger.info(
                    "Metal: chunked prefill enabled (paged attention), "
                    "max_num_batched_tokens=%d",
                    scheduler_config.max_num_batched_tokens,
                )
            else:
                # The non-paged MLX path does not honor chunked-prefill
                # scheduler boundaries.  Disable so the scheduler only
                # requests full prefills.
                scheduler_config.enable_chunked_prefill = False

                # Without chunked prefill, the scheduler must fit the
                # entire prompt in a single step.  Ensure
                # max_num_batched_tokens (and max_num_scheduled_tokens)
                # are at least max_model_len; otherwise the scheduler
                # silently refuses to schedule any prompt that exceeds
                # the budget.
                if model_config is not None:
                    model_max = model_config.max_model_len
                    if scheduler_config.max_num_batched_tokens < model_max:
                        scheduler_config.max_num_batched_tokens = model_max
                    if (
                        scheduler_config.max_num_scheduled_tokens is not None
                        and scheduler_config.max_num_scheduled_tokens < model_max
                    ):
                        scheduler_config.max_num_scheduled_tokens = model_max

                logger.info(
                    "Metal: disabled chunked prefill (non-paged path), "
                    "max_num_batched_tokens=%d",
                    scheduler_config.max_num_batched_tokens,
                )

        # Disable cascade attention (not supported), then let the adapter
        # apply any model-specific normalisations (e.g. clearing
        # ``multimodal_config`` for model types served on the text-only
        # backbone — see ``DefaultModelAdapter.normalize_model_config``).
        if model_config is not None:
            model_config.disable_cascade_attn = True
            from aphrodite.metal.v1.model_adapter import DefaultModelAdapter

            DefaultModelAdapter().normalize_model_config(model_config)

        # STT model detection — set tokenizer fallback if not already configured.
        # Lazy imports to avoid circular import: platform.py is loaded during
        # aphrodite.config init, and stt.detection imports from aphrodite.config.
        from aphrodite.metal.stt.detection import is_stt_model
        from aphrodite.metal.stt.policy import apply_stt_scheduler_policy
        from aphrodite.metal.utils import get_model_download_path

        resolved_model = (
            get_model_download_path(model_config.model)
            if model_config is not None
            else None
        )
        if resolved_model is not None and is_stt_model(resolved_model):
            was_async_scheduling = bool(scheduler_config.async_scheduling)
            apply_stt_scheduler_policy(model_config, scheduler_config)
            if was_async_scheduling and not scheduler_config.async_scheduling:
                logger.info("STT: disabled async_scheduling")
            logger.info("STT model detected")

        # Log memory configuration
        total_mem = cls.get_device_total_memory()
        available_mem = cls.get_device_available_memory()
        logger.info(
            f"Metal memory: {total_mem / 1e9:.1f}GB total, "
            f"{available_mem / 1e9:.1f}GB available"
        )

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        """Metal supports hybrid KV cache for models like Qwen3.5 (SDPA + GDN)."""
        return True

    @classmethod
    def _find_non_ssm_backend(
        cls, aphrodite_config: "AphroditeConfig"
    ) -> "type[AttentionBackend] | None":
        """Return a Metal-specific backend for block_size calculation.

        Since MLX models don't populate static_forward_context, the default
        Platform._find_non_ssm_backend (which walks attention layers via
        get_layers_from_aphrodite_config) returns nothing. We override to return
        the synthetic MetalBackend, which advertises Metal's MultipleOf(16)
        kernel alignment to the framework's hybrid-block-size math.
        """
        from aphrodite.metal.metal_backend import MetalBackend

        return MetalBackend

    @classmethod
    def update_block_size_for_backend(cls, aphrodite_config: "AphroditeConfig") -> None:
        """Update block_size for Metal platform.

        Delegates to Aphrodite's base implementation, which reads the Metal kernel
        alignment (MultipleOf(16)) from our :meth:`_find_non_ssm_backend`
        override. Adds a one-time warning when paged attention is enabled for
        a hybrid model, explaining the cache-block-size translation mechanism
        (PR #235).
        """
        from aphrodite.metal.config import get_config

        metal_config = get_config()
        model_config = aphrodite_config.model_config

        if not model_config:
            return

        # For hybrid models with paged attention, log a warning explaining the
        # block-size translation mechanism.
        #
        # Background:
        # - Aphrodite requires block_size=160 (or larger) for hybrid models to satisfy
        #   page size divisibility validation between SDPA and Mamba layers.
        #
        # Solution:
        # - Aphrodite sees a large block_size (e.g., 144 = 16 * 9) for its scheduler
        #   validation.
        # - The Metal kernel uses a translated block_size (16, the kernel sweet
        #   spot) that it supports.
        # - Each Aphrodite block is split into ratio = cache_block_size / kernel_block_size
        #   kernel blocks. For example, one Aphrodite block of 144 tokens becomes 9 kernel
        #   blocks of 16 tokens each.
        # - The KV cache is reshaped (zero-copy) to match: [num_blocks, 144, ...] →
        #   [num_blocks*9, 16, ...]. The physical memory layout is unchanged.
        # - Block tables are expanded so the kernel reads the correct blocks.
        #
        # This is a logical transformation only — the computation is identical, just
        # the kernel sees more, smaller blocks.
        if model_config.is_hybrid and metal_config.use_paged_attention:
            logger.warning(
                "Hybrid model (e.g., Qwen3.5) with paged attention enabled. "
                "Using block-size translation (PR #235) to convert Aphrodite's large "
                "block_size to a Metal kernel-compatible size.\n"
                "  Mechanism: Each Aphrodite block is split into multiple kernel blocks.\n"
                "  Example: Aphrodite block_size=144 → kernel block_size=16 (ratio=9).\n"
                "  The KV cache is reshaped (zero-copy) and block tables are expanded.\n"
                "  This is a logical transformation — physical memory is unchanged."
            )

        # Delegate the rest to upstream. With our ``_find_non_ssm_backend``
        # returning :class:`MetalBackend` (which advertises ``MultipleOf(16)``),
        # Aphrodite's Phase 1 picks a kernel-aligned default of 16 for non-hybrid
        # models (matching the kernel sweet spot), and Phase 2
        # (``_align_hybrid_block_size``) handles hybrid alignment. The kernel
        # layer (``_pick_kernel_block_size``) validates the final
        # ``block_size`` at request time.
        super().update_block_size_for_backend(aphrodite_config)

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        """Get the attention backend class for Metal."""
        from aphrodite.v1.attention.backends.registry import AttentionBackendEnum

        if selected_backend and selected_backend != AttentionBackendEnum.CPU_ATTN:
            logger.info(f"Cannot use {selected_backend} backend on Metal/MLX.")
        if attn_selector_config.use_mla:
            # MLA attention is handled by the aphrodite metal model runner (MLAPagedAttentionWrapper),
            # not by Aphrodite's attention backend selector. Continue to return CPU_ATTN below.
            logger.info(
                "MLA model detected; attention handled by aphrodite metal model runner"
            )
        if attn_selector_config.use_sparse:
            raise NotImplementedError("Sparse Attention is not supported on Metal/MLX.")
        return AttentionBackendEnum.CPU_ATTN.get_path()

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        """Verify that quantization method is supported.

        Args:
            quant: Quantization method name

        Raises:
            ValueError: If quantization is not supported
        """
        # Allow all quantization methods to pass through - actual support
        # depends on the model implementation. This avoids blocking models
        # that use quantization formats we may be able to handle.
        pass

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """Check if pin_memory is available for Metal platform.

        Returns:
            False - pin_memory is not needed/supported on Metal/MLX

        Note:
            Although MLX uses unified memory (which theoretically could benefit
            from pin_memory), we disable it because:
            1. PyTorch's pin_memory is primarily designed for CUDA
            2. In our architecture, PyTorch tensors are on CPU for MLX interop
            3. pin_memory on CPU can cause issues or errors
            4. Unified memory already provides fast CPU-GPU transfers without pinning
        """
        return False
