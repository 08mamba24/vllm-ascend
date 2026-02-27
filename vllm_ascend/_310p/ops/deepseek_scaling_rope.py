#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

import math
import os

import torch
import torch_npu
from vllm.logger import logger
from vllm_ascend._310p.ops.rotary_embedding import (
    _record_cos_and_sin_cache_310p,
    _rope_forward_oot,
)
from vllm_ascend.ops.rotary_embedding import AscendDeepseekScalingRotaryEmbedding


class AscendDeepseekScalingRotaryEmbedding310(AscendDeepseekScalingRotaryEmbedding):
    """
    310P-specific DeepseekScalingRotaryEmbedding that uses correct cache format for MLA.

    The key difference from the base AscendDeepseekScalingRotaryEmbedding is that we use
    _record_cos_and_sin_cache_310p which correctly splits cos_sin_cache
    into [L, D] format instead of the erroneous format produced by the base class's
    _set_cos_sin_cache method.

    This ensures that get_cos_and_sin_mla (used by 310P MLA) receives
    properly shaped cos/sin caches.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ) -> None:
        # Initialize parent's scaling-related attributes
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = float(
            self._yarn_get_mscale(self.scaling_factor, float(mscale))
            / self._yarn_get_mscale(self.scaling_factor, float(mscale_all_dim))
            * attn_factor
        )

        # Call grandparent's __init__ to avoid base AscendDeepseekScalingRotaryEmbedding's
        # _set_cos_sin_cache which produces incorrect cache format
        # NOTE: GLM-4.7-Flash uses DeepSeekV2MLAAttention (MLA), NOT DeepSeek scaling
        # DeepSeek scaling is not used for MLA models, skip this initialization
        # The 310P MLA path will use get_cos_and_sin_mla() which expects global _cos_cache and _sin_cache
        # Those globals are initialized by set_cos_and_sin() in rotary_embedding.py
        logger.info("[310P DeepSeek Scaling RoPE] Skipping cache init - MLA uses global cache from set_cos_and_sin()")
        # No cache initialization for DeepSeek scaling (not needed for MLA)

    def _yarn_get_mscale(self, scale: float = 1, mscale: float = 1) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: torch.Tensor | None = None,
        is_neox_style_override: bool | None = None,
    ):
        is_neox_style = self.is_neox_style
        if is_neox_style_override is not None:
            is_neox_style = is_neox_style_override

        # Note: we implement the non neox_style method with shuffle the last dim and neox style
        # calculation method which is also more compute friendly to the ascend machine
        # https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/blob/main/modeling_deepseek.py
        is_neox_style_npu = True
        if is_neox_style is False:
            b, h_q, d = query.shape
            query = query.view(b, h_q, d // 2, 2).transpose(3, 2).reshape(b, h_q, d)
            b, h_k, d = key.shape
            key = key.view(b, h_k, d // 2, 2).transpose(3, 2).reshape(b, h_k, d)

        # Use 310P-specific RoPE forward implementation
        return _rope_forward_oot(self, positions, query, key, is_neox_style_npu, offsets)
