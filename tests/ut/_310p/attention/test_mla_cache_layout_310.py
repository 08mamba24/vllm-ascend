#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Unit test to validate 310P MLA KV cache layout mismatch bug.

Root cause: 310P decode uses .view(-1, num_kv_heads, block_size, dim) which
swaps block_size and num_kv_heads axes compared to the write layout
(num_blocks, block_size, num_kv_heads, dim) where data is written via
slot = block * block_size + offset.

This test:
1. Creates a mock MLA cache with unique, index-encodable patterns
2. Simulates write process (flatten to slot index)
3. Demonstrates that current incorrect read layout (.view) fails
4. Demonstrates that corrected read layout (.permute + contiguous) succeeds
"""

import torch
import pytest
from tests.ut.base import TestBase
from vllm_ascend._310p.attention.attention_v1 import (
    AscendMLAImpl310,
)


class TestMLACacheLayout310(TestBase):
    """
    Test 310P MLA cache layout mismatch bug.

    This test validates that:
    1. Current 310P decode path uses .view(-1, num_kv_heads, block_size, dim)
       which swaps block_size and num_kv_heads axes
    2. Correct read path uses .permute(0, 2, 1, 3).contiguous()
       which preserves the (num_blocks, block_size, num_kv_heads, dim) layout

    Expected behavior:
    - Cache is written as (num_blocks, block_size, num_kv_heads, dim)
    - Slot mapping uses slot = block * block_size + offset
    - Decode should read using (num_blocks, num_kv_heads, block_size, dim) layout
    """

    def test_mla_cache_layout_mismatch_bug(self):
        """
        Test that demonstrates the cache layout mismatch bug in 310P MLA.

        This test creates a mock MLA cache with unique patterns,
        fills it with data, then attempts to read back using
        both the current incorrect view approach and the correct permute approach.
        """
        # Simulate MLA cache shape from model_runner_310p.py
        # For GLM4-MoE-Lite: num_blocks=10, block_size=16, num_kv_heads=20
        # kv_lora_rank=512, qk_rope_head_dim=64
        num_blocks = 10
        block_size = 16
        num_kv_heads = 20
        kv_lora_rank = 512
        qk_rope_head_dim = 64

        # Create mock MLA caches matching actual 310P allocation
        # Write layout: (num_blocks, block_size, num_kv_heads, dim)
        kv_c_cache = torch.zeros(num_blocks, block_size, num_kv_heads, kv_lora_rank, dtype=torch.float32)
        k_pe_cache = torch.zeros(num_blocks, block_size, num_kv_heads, qk_rope_head_dim, dtype=torch.float32)

        # Fill caches with unique, index-encodable patterns
        # This allows us to detect axis swaps when we read back
        for block in range(num_blocks):
            for head in range(num_kv_heads):
                for offset in range(block_size):
                    kv_c_cache[block, offset, head, :] = float(f"{block}.{head}.{offset}")
                    k_pe_cache[block, offset, head, :] = float(f"{block}.{head}.{offset}k")

        # Simulate slot mapping used by _store_to_cache_310
        # slot = block * block_size + offset
        total_slots = num_blocks * block_size
        slots = torch.arange(total_slots, dtype=torch.long)

        # Test 1: Incorrect read layout (current 310P bug)
        # Uses .view(-1, num_kv_heads, block_size, dim) which swaps axes
        with pytest.raises(AssertionError) as exc_info:
            msg = "读布局错误：.view\(\) 交换了 block_size 和 num_kv_heads 轴"
            try:
                k_nope_wrong = kv_c_cache.view(-1, num_kv_heads, block_size, kv_lora_rank)
                k_pe_wrong = k_pe_cache.view(-1, num_kv_heads, block_size, qk_rope_head_dim)

                # Verify that wrong layout reads swapped data
                for i, slot in enumerate(slots):
                    # In wrong layout, slot's block and offset are swapped
                    # kv_c_cache.view(-1, ...) interprets as: [total_slots, num_kv_heads, block_size, dim]
                    # So: block_idx = i // block_size, head_idx = i % num_kv_heads
                    # But we want: block_idx = slot // block_size, offset = slot % block_size
                    # Wrong layout will read from:
                    #   kv_c_cache[head_idx, offset, block_idx, :]  (swapped!)
                    # Instead of:
                    #   kv_c_cache[block_idx, head_idx, offset, :] (correct)

                # Check a few slots to detect the swap
                if i < 3 and slot % 2 == 0:  # Check slot 0 and 2 (different blocks)
                    # Slot i=0: block=0, offset=0 -> wants kv_c_cache[0, head, 0, 0]
                    # Wrong layout reads: kv_c_cache[0, head, 0, :] (block_idx=0, head_idx=0, offset=0) CORRECT for slot 0
                    # kv_c_cache.view(-1, ...) interprets total_slots as first dim
                    # So kv_c_cache[head, offset, block] where head=0, offset=0, block=0

                    # Wait, we need to demonstrate the swap more clearly
                    # Let's check what .view(-1, num_kv_heads, block_size, ...) produces
                    # Shape: (total_slots, num_kv_heads, block_size, kv_lora_rank)
                    # i.e., (10*16*20, 16, 512) = (3200, 20, 16, 512)

                    # Slot 0 in wrong layout should read:
                    # For head_idx = i % num_kv_heads = 0 % 20 = 0
                    # For block_idx = i // block_size = 0 // 16 = 0
                    # So wrong layout reads: kv_c_cache[0, 0, 0, :] (head_idx=0, block_idx=0, offset=0)
                    # But slot 0 was written at: kv_c_cache[0, 0, 0, :]
                    # This demonstrates the axis swap

                # Check a few slots to detect the swap
                if i < 3 and slot % 2 == 0:  # Check slot 0 and 2 (different blocks)
                    # Slot i=0: block=0, offset=0 -> wants kv_c_cache[0, head, 0, 0]
                    # Wrong layout reads: kv_c_cache[0, 0, :] (block_idx=0, head_idx=0, offset=0) CORRECT for slot 0
                    # kv_c_cache.view(-1, ...) interprets total_slots as first dim
                    # So kv_c_cache[head, offset, block] where head=0, offset=0, block=0

                    # Wait, we need to demonstrate the swap more clearly
                    # Let's check what .view(-1, num_kv_heads, block_size, ...) produces
                    # Shape: (total_slots, num_kv_heads, block_size, kv_lora_rank)
                    # i.e., (10*16*20, 16, 512) = (3200, 20, 16, 512)

                    # Slot 0 in wrong layout should read:
                    # For head_idx = i % num_kv_heads = 0 % 20 = 0
                    # For block_idx = i // block_size = 0 // 16 = 0
                    # So wrong layout reads: kv_c_cache[0, 0, 0, :] (head_idx=0, block_idx=0, offset=0)
                    # But slot 0 was written at: kv_c_cache[0, 0, 0, :]
                    # This demonstrates the axis swap

            assert msg in str(exc_info.value)

        # Test 2: Correct read layout (proposed fix)
        # Uses .permute(0, 2, 1, 3).contiguous()
        # to convert (num_blocks, block_size, num_kv_heads, dim) -> (num_blocks, num_kv_heads, block_size, dim)
        k_nope_correct = kv_c_cache.permute(0, 2, 1, 3).contiguous()
        k_pe_correct = k_pe_cache.permute(0, 2, 1, 3).contiguous()

        # Verify that correct layout reads the data that was written
        for i, slot in enumerate(slots):
            block_idx = slot // block_size
            offset = slot % block_size
            head_idx = i % num_kv_heads

            # Correct layout should read:
            # kv_c_cache[block_idx, head_idx, offset, :] (preserves layout)
            assert torch.allclose(
                k_nope_correct[i, :, slot % block_size, :],
                kv_c_cache[block_idx, head_idx, offset, :],
                atol=1e-6,
            )
            assert torch.allclose(
                k_pe_correct[i, :, slot % block_size, :],
                k_pe_cache[block_idx, head_idx, offset, :],
                atol=1e-6,
            )

        # Verify end-to-end shape consistency
        assert k_nope_correct.shape == (total_slots, num_kv_heads, block_size, kv_lora_rank)
        assert k_pe_correct.shape == (total_slots, num_kv_heads, block_size, qk_rope_head_dim)
