# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from sonar._custom_ops import xqa_paged_attention


def print_cuda_info():
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
    print(f"CUDA Version: {torch.version.cuda}")


def reset_cuda():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.init()


def test_single_config(batch_size=1, num_heads=32, num_kv_heads=4, head_size=128, block_size=16, max_seq_len=128):
    """Test a single XQA configuration with proper cleanup"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for XQA paged attention")

    print("\nTesting configuration:")
    print(f"  batch_size: {batch_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  head_size: {head_size}")
    print(f"  block_size: {block_size}")
    print(f"  max_seq_len: {max_seq_len}")

    reset_cuda()
    print(f"CUDA memory after reset: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    rotary_embedding_dim = head_size // 2
    scale = 1.0 / (head_size**0.5)

    query = torch.randn(batch_size, num_heads, head_size, dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()

    num_blocks = (max_seq_len + block_size - 1) // block_size * batch_size
    kv_cache = torch.randn(num_blocks, num_kv_heads, block_size, 2, head_size, dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()

    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device="cuda")
    block_tables = block_tables.reshape(batch_size, max_blocks_per_seq)
    torch.cuda.synchronize()

    seq_lens = torch.full((batch_size,), max_seq_len, dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()

    out = torch.empty_like(query)
    torch.cuda.synchronize()

    print("\nTensor shapes:")
    print(f"  query: {query.shape}")
    print(f"  kv_cache: {kv_cache.shape}")
    print(f"  block_tables: {block_tables.shape}")
    print(f"  seq_lens: {seq_lens.shape}")

    try:
        xqa_paged_attention(
            out=out,
            query=query,
            kv_cache=kv_cache,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rotary_embedding_dim=rotary_embedding_dim,
            scale=scale,
            block_tables=block_tables,
            seq_lens=seq_lens,
            block_size=block_size,
            max_seq_len=max_seq_len,
            kv_cache_dtype="auto",
            k_scale=1.0,
            v_scale=1.0,
        )
        torch.cuda.synchronize()
        print("✓ Configuration succeeded")
        return True
    except Exception as e:
        print(f"✗ Configuration failed: {str(e)}")
        return False
    finally:
        reset_cuda()


if __name__ == "__main__":
    import os

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Known working config
    test_single_config(batch_size=1, num_heads=32, num_kv_heads=4, head_size=128, block_size=16, max_seq_len=128)
