# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused checks for the local GLM DFlash2 compatibility patch.

Run with the Sonar venv; these checks use CPU tensors and no model weights.
"""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sonar.model_executor.models.interfaces import supports_eagle3
from sonar.models.glm5next.nvidia import model as glm
from sonar.v1.core import kv_cache_utils as kv
from sonar.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from sonar.v1.kv_cache_interface import (
    KpoolTailSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    create_kv_cache_views,
)
from sonar.v1.worker.gpu.spec_decode.eagle.eagle3_utils import set_eagle3_aux_hidden_state_layers


class GlmAuxCaptureTest(unittest.TestCase):
    def make_model(self, layers):
        model = object.__new__(glm.Glm5NextModel)
        torch.nn.Module.__init__(model)
        model.start_layer = 0
        model.end_layer = len(layers)
        model._active_layers = layers
        model.layers = layers
        model.is_sequence_parallel = False
        model.norm = lambda x: x * 2
        return model

    def forward(self, model):
        with patch.object(
            glm,
            "get_pp_group",
            return_value=SimpleNamespace(is_first_rank=True, is_last_rank=True),
        ):
            return model.forward(None, torch.tensor([0]), None, torch.tensor([[1.0, 2.0]]))

    def test_plain_layer_capture_does_not_add_residual_twice(self):
        layer = Mock(
            return_value=(
                torch.tensor([[3.0, 4.0]]),
                torch.tensor([[10.0, 20.0]]),
                None,
                None,
            )
        )
        model = self.make_model([layer])
        baseline = self.forward(model)
        model._set_aux_hidden_state_layers((0, 1))
        output, aux = self.forward(model)
        torch.testing.assert_close(output, baseline, rtol=0, atol=0)
        torch.testing.assert_close(aux[0], torch.tensor([[1.0, 2.0]]))
        torch.testing.assert_close(aux[1], torch.tensor([[3.0, 4.0]]))

    def test_mhc_capture_materializes_streams_without_changing_target_state(self):
        hidden = torch.tensor([[3.0, 4.0]])
        residual = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
        post = torch.tensor([[0.5, 1.5]])
        comb = torch.eye(2).unsqueeze(0)
        originals = [x.clone() for x in (hidden, residual, post, comb)]
        first = Mock(return_value=(hidden, residual, post, comb))
        first.n = 2
        first.hc_post = Mock(
            side_effect=lambda h, r, p, c: h[:, None] * p[:, :, None] + torch.einsum("bij,bih->bjh", c, r)
        )

        def finish(positions, h, r, p, c):
            for got, expected in zip((h, r, p, c), (hidden, residual, post, comb)):
                self.assertIs(got, expected)
            return h + r.mean(1), None, None, None

        model = self.make_model([first, Mock(side_effect=finish)])
        baseline = self.forward(model)
        first.hc_post.assert_not_called()
        model._set_aux_hidden_state_layers((1, 2))
        output, aux = self.forward(model)
        torch.testing.assert_close(output, baseline, rtol=0, atol=0)
        torch.testing.assert_close(aux[0], torch.tensor([[9.0, 11.0]]))
        torch.testing.assert_close(aux[1], baseline / 2)
        for got, original in zip((hidden, residual, post, comb), originals):
            torch.testing.assert_close(got, original, rtol=0, atol=0)

    def test_multimodal_wrapper_configures_checkpoint_layers(self):
        self.assertTrue(supports_eagle3(glm.Glm5NextForConditionalGeneration))
        self.assertTrue(supports_eagle3(glm.Glm5NextForCausalLM))
        target = object.__new__(glm.Glm5NextForConditionalGeneration)
        torch.nn.Module.__init__(target)
        model = self.make_model([None] * 45)
        target.language_model = SimpleNamespace(
            embed_input_ids=lambda _: None,
            forward=lambda input_ids, positions: None,
            model=model,
        )
        target._language_model_names = ["language_model"]
        config = SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(dflash_config={"target_layer_ids": [5, 14, 24, 33, 42]})
            )
        )
        set_eagle3_aux_hidden_state_layers(target, config)
        self.assertEqual(model.aux_hidden_state_layers, (6, 15, 25, 34, 43))

    def test_sequence_parallel_capture_gathers_and_removes_padding(self):
        model = self.make_model([Mock(side_effect=lambda p, h, r, post, comb: (h + 1, None, None, None))])
        model.is_sequence_parallel = True
        model._set_aux_hidden_state_layers((0, 1))
        with (
            patch.object(glm, "sp_shard", side_effect=lambda h: h),
            patch.object(glm, "sp_all_gather", side_effect=lambda h: torch.cat([h, torch.zeros_like(h)])) as gather,
        ):
            output, aux = self.forward(model)
        self.assertEqual(gather.call_count, 2)
        torch.testing.assert_close(aux[0], torch.tensor([[1.0, 2.0]]))
        torch.testing.assert_close(aux[1], torch.tensor([[2.0, 3.0]]))
        torch.testing.assert_close(output, torch.tensor([[4.0, 6.0]]))

    def test_sliding_draft_cache_aliasing_and_accounting(self):
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=1, decode_context_parallel_size=1),
            model_config=SimpleNamespace(max_model_len=16384),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
            cache_config=SimpleNamespace(
                num_gpu_blocks_override=None,
                prefix_cache_retention_interval=None,
                mamba_cache_mode="align",
            ),
            max_in_flight_tokens=16384,
            speculative_config=SimpleNamespace(use_eagle_block_drop=lambda: True),
        )
        specs = {}
        for i in range(2):
            specs[f"mla{i}"] = MLAAttentionSpec(block_size=2304, num_kv_heads=1, head_size=576, dtype=torch.uint8)
            specs[f"idx{i}"] = MLAAttentionSpec(
                block_size=2304,
                num_kv_heads=1,
                head_size=132,
                dtype=torch.uint8,
                tokens_per_state=4,
            )
            specs[f"tail{i}"] = KpoolTailSpec(
                block_size=4,
                num_kv_heads=2,
                head_size=128,
                head_size_v=0,
                dtype=torch.bfloat16,
                sliding_window=4,
            )
        for i in range(4):
            specs[f"mamba{i}"] = MambaSpec(
                block_size=2304,
                shapes=((16, 128, 128),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            )
        target_groups = kv.get_kv_cache_groups(config, specs)
        block_bytes = kv._get_kv_cache_bytes_per_block(target_groups)
        for i in range(3):
            specs[f"draft{i}"] = SlidingWindowSpec(
                block_size=16,
                num_kv_heads=2,
                head_size=128,
                dtype=torch.uint8,
                sliding_window=2048,
            )
        groups = kv.get_kv_cache_groups(config, specs)
        draft_groups = [g for g in groups if g.is_eagle_group]
        self.assertEqual([len(g.layer_names) for g in draft_groups], [2, 1])
        self.assertEqual(kv._get_kv_cache_bytes_per_block(groups), block_bytes)
        cache = kv.get_kv_cache_config_from_groups(config, groups, block_bytes * 10)
        self.assertEqual(cache.num_blocks, 10)
        tensors = {name: t for t in cache.kv_cache_tensors for name in t.layers}
        self.assertEqual(set(tensors), set(specs))
        self.assertEqual({t.size for t in tensors.values()}, {block_bytes * 10})
        for group in draft_groups:
            for i, name in enumerate(group.layer_names):
                self.assertEqual(tensors[name].offset, tensors[f"mla{i}"].offset)
                self.assertEqual(tensors[name].block_stride, specs["mla0"].page_size_bytes)
                self.assertEqual(
                    replace(group.kv_cache_spec, page_size_padded=None),
                    replace(specs[name], block_size=2304),
                )
        for i in range(2):
            self.assertEqual(tensors[f"tail{i}"].offset, tensors[f"idx{i}"].offset)
        raw = torch.zeros(block_bytes * 10, dtype=torch.uint8)
        draft_spec = draft_groups[0].kv_cache_spec
        view = create_kv_cache_views(
            raw,
            draft_spec,
            10,
            KVCacheLayout.LBHNC,
            tensors["draft0"],
            kernel_block_size=draft_spec.block_size,
        )[0]
        view[3].fill_(1)
        start = tensors["draft0"].offset + 3 * tensors["draft0"].block_stride
        end = start + replace(specs["draft0"], block_size=2304).page_size_bytes
        self.assertEqual(raw[:start].count_nonzero().item(), 0)
        self.assertEqual(raw[start:end].count_nonzero().item(), end - start)
        self.assertEqual(raw[end:].count_nonzero().item(), 0)
        extra_blocks = sum(
            g.kv_cache_spec.max_memory_usage_bytes(config) // g.kv_cache_spec.page_size_bytes for g in draft_groups
        )
        self.assertEqual(
            kv._max_memory_usage_bytes_from_groups(config, groups),
            kv._max_memory_usage_bytes_from_groups(config, target_groups) + extra_blocks * block_bytes,
        )

        config.speculative_config.use_eagle_block_drop = lambda: False
        self.assertFalse(any(g.is_eagle_group for g in kv.get_kv_cache_groups(config, specs)))

        # A draft that cannot fit after logical-span alignment must use the generic layout.
        oversized = dict(specs)
        oversized["draft0"] = replace(specs["draft0"], num_kv_heads=4)
        self.assertIsNone(kv._get_kv_cache_groups_glm5_next(config, oversized))
        oversized["draft0"] = replace(specs["draft0"], block_size=16384)
        self.assertIsNone(kv._get_kv_cache_groups_glm5_next(config, oversized))

    def test_dflash_block_drop_can_erase_a_reusable_mamba_checkpoint(self):
        """Unshifted DFlash KV needs no EAGLE lookahead beyond the shared prefix."""
        block_size = 2304
        specs = [
            MLAAttentionSpec(block_size=block_size, num_kv_heads=1, head_size=576, dtype=torch.uint8),
            MambaSpec(
                block_size=block_size,
                shapes=((2, 2),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            ),
            SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=2,
                head_size=128,
                dtype=torch.uint8,
                sliding_window=2048,
            ),
        ]
        for drop, expected in [(True, 0), (False, 9216)]:
            with self.subTest(drop=drop):
                groups = [
                    KVCacheGroupSpec([str(i)], spec, is_eagle_group=drop and i == 2) for i, spec in enumerate(specs)
                ]
                coordinator = HybridKVCacheCoordinator(
                    KVCacheConfig(32, [], groups),
                    max_model_len=16384,
                    max_in_flight_tokens=16384,
                    use_eagle=drop,
                    enable_caching=True,
                    enable_kv_cache_events=False,
                    dcp_world_size=1,
                    pcp_world_size=1,
                    scheduler_block_size=block_size,
                    hash_block_size=block_size,
                )
                hashes = [bytes([i]) * 32 for i in range(4)]
                pool = coordinator.block_pool
                # Full target KV, one saved recurrent state at token 9216,
                # and a valid drafter sliding-window tail at that boundary.
                for gid, indices in [(0, range(4)), (1, [3]), (2, [2, 3])]:
                    blocks = [pool.null_block] * 4
                    for i in indices:
                        blocks[i] = pool.get_new_blocks(1)[0]
                    pool.cache_full_blocks(
                        SimpleNamespace(block_hashes=hashes),
                        blocks,
                        0,
                        4,
                        block_size,
                        gid,
                    )
                _, hit, _ = coordinator.find_longest_cache_hit(hashes, 10239)
                self.assertEqual(hit, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
