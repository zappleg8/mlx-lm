# Copyright © 2026 Apple Inc.
"""Zyphra ZAYA1 (MLX port).

Hybrid architecture: 80 layers strictly alternating between CCA attention
(Compressed Causal Attention with depthwise 1D conv on Q+K and a time-shifted
V stream) and MoE (top-1 routing with optional MoD skip-expert and EDA).

This file is structured to mirror the PyTorch reference at
`Zyphra/transformers @ zaya1` (modular_zaya.py) at the parameter level,
so HF safetensors weights load with strict=True via Model.sanitize.
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "zaya"
    hidden_size: int = 2048
    num_hidden_layers: int = 80
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    num_query_groups: int = 2
    cca_num_q_heads: int = 8
    cca_time0: int = 2
    cca_time1: int = 2
    ffn_hidden_size: int = 4096
    num_experts: int = 16
    moe_router_topk: int = 1
    zaya_mlp_expansion: int = 256
    zaya_use_mod: bool = True
    zaya_use_eda: bool = True
    vocab_size: int = 262272
    max_position_embeddings: int = 131072
    partial_rotary_factor: float = 0.5
    rope_theta: float = 5000000.0
    rope_scaling: Optional[dict] = None
    norm_epsilon: float = 1e-5
    attention_bias: bool = False
    lm_head_bias: bool = False
    add_bias_linear: bool = False
    tie_word_embeddings: bool = True
    residual_in_fp32: bool = True
    scale_residual_merge: bool = True
    activation_func: str = "swiglu"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


# Hardcoded in modular_zaya.py:1089. EDA is gated off for the first MoE layer
# in the global decoder layer index (which is layer 1 since layer 0 is ATT).
ZAYA_FIRST_MOE_LAYER = 1


class ResidualScaling(nn.Module):
    """Per-feature affine on residual streams before merging.

    Per modular_zaya.py:1003-1033. Layer 0 only has hidden_states_*
    parameters; non-first layers also have residual_*.
    """

    def __init__(self, args: ModelArgs, layer_n: int):
        super().__init__()
        self.not_first_layer = layer_n != 0
        self.hidden_states_scale = mx.ones((args.hidden_size,))
        self.hidden_states_bias = mx.zeros((args.hidden_size,))
        if self.not_first_layer:
            self.residual_scale = mx.ones((args.hidden_size,))
            self.residual_bias = mx.zeros((args.hidden_size,))

    def __call__(self, residual, hidden_states):
        hidden_states = (hidden_states + self.hidden_states_bias) * self.hidden_states_scale
        if self.not_first_layer:
            residual = (residual + self.residual_bias) * self.residual_scale
        return residual, hidden_states


class MLP(nn.Module):
    """Single SwiGLU expert. Per modular_zaya.py:1190-1275.

    Note: ffn_hidden_size_out is ffn_hidden_size // 2 due to gated linear unit.
    add_bias_linear is False per config, so no bias on either Linear.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        ffn_out = args.ffn_hidden_size // 2  # gated linear unit halves output
        self.linear_fc1 = nn.Linear(args.hidden_size, args.ffn_hidden_size, bias=args.add_bias_linear)
        self.linear_fc2 = nn.Linear(ffn_out, args.hidden_size, bias=args.add_bias_linear)

    def __call__(self, x):
        h = self.linear_fc1(x)
        a, b = mx.split(h, 2, axis=-1)
        return self.linear_fc2(nn.silu(a) * b)


class SequentialMLP(nn.Module):
    """Container of MoE experts. Per modular_zaya.py:1278-1326.

    Holds a list of MLP modules. Forward routing logic is implemented in
    ZayaBlock — at the skeleton stage we just declare the parameters.
    """

    def __init__(self, args: ModelArgs, num_local_experts: int):
        super().__init__()
        self.local_experts = [MLP(args) for _ in range(num_local_experts)]


class CCA(nn.Module):
    """Compressed Causal Attention. Per modular_zaya.py:285-521.

    Replaces standard QKV with: linear projections + two-stage depthwise 1D
    causal conv on concatenated Q+K + L2-normalized Q/K with per-KV-head temp +
    two-stream V (current + time-shifted hidden state).

    Skeleton only — forward will be implemented in Phase 3.
    """

    def __init__(self, args: ModelArgs, layer_number: int):
        super().__init__()
        self.layer_number = layer_number
        self.hidden_size = args.hidden_size
        self.num_kv_heads = args.num_query_groups  # 2
        self.num_q_heads = args.cca_num_q_heads  # 8
        self.num_heads = args.num_attention_heads  # 16 (for head_dim calc)
        self.head_dim = args.hidden_size // self.num_heads  # 128
        self.latent_k_dim = self.num_kv_heads * self.head_dim  # 256
        self.latent_q_dim = self.num_q_heads * self.head_dim  # 1024
        self.cca_time0 = args.cca_time0
        self.cca_time1 = args.cca_time1

        self.linear_q = nn.Linear(self.hidden_size, self.latent_q_dim, bias=args.attention_bias)
        self.linear_k = nn.Linear(self.hidden_size, self.latent_k_dim, bias=args.attention_bias)
        self.val_proj1 = nn.Linear(self.hidden_size, self.latent_k_dim // 2, bias=args.attention_bias)
        self.val_proj2 = nn.Linear(self.hidden_size, self.latent_k_dim // 2, bias=args.attention_bias)

        in_out_ch = self.latent_k_dim + self.latent_q_dim  # 1280
        self.conv_qk = nn.Sequential(
            nn.Conv1d(
                in_channels=in_out_ch,
                out_channels=in_out_ch,
                kernel_size=self.cca_time0,
                groups=in_out_ch,  # depthwise: groups = in_ch = out_ch
                padding=0,
                stride=1,
                bias=True,
            ),
            nn.Conv1d(
                in_channels=in_out_ch,
                out_channels=in_out_ch,
                kernel_size=self.cca_time1,
                groups=(self.num_kv_heads + self.num_q_heads),  # 10
                padding=0,
                stride=1,
                bias=True,
            ),
        )
        # Per-KV-head learnable temperature
        self.temp = mx.zeros((self.num_kv_heads,))

    def __call__(
        self,
        hidden_states: mx.array,
        cca_mask: Optional[mx.array] = None,
        past_key_values=None,
    ):
        """Compressed Causal Attention forward.

        Args:
          hidden_states: (B, S, H) input from the layer's input_norm.
          cca_mask: optional (B, S) attention mask; multiplied into hidden_states
            during prefill. If None, treated as all-ones (no masking).
          past_key_values: ZayaDynamicCache for generation. If None, prefill
            mode only — no conv state cache, no prev_hs cache.

        Returns: (Q, K, V) flattened to HF layout (B, S, n_heads * head_dim).
          Q: (B, S, num_q_heads * head_dim) = (B, S, 1024)
          K: (B, S, num_kv_heads * head_dim) = (B, S, 256)
          V: (B, S, num_kv_heads * head_dim) = (B, S, 256)

        Implements modular_zaya.py:370-521 in HF (B, S, H) layout.
        Generation-with-cache (`has_previous_state`) is deferred to Phase 4.
        """
        B, S, _ = hidden_states.shape
        gqa_groups = self.num_q_heads // self.num_kv_heads  # 4
        sqrt_head_dim = mx.array(self.head_dim ** 0.5, dtype=hidden_states.dtype)

        # Apply cca_mask during prefill if S > 1 and mask is provided.
        if cca_mask is not None and S > 1:
            hidden_states = hidden_states * cca_mask[:, :, None]

        # ---- Linear projections ----
        q = self.linear_q(hidden_states)  # (B, S, 1024)
        k = self.linear_k(hidden_states)  # (B, S, 256)
        qk_packed0 = mx.concatenate([q, k], axis=-1)  # (B, S, 1280)

        # ---- Pre-conv mean residual ----
        query_pre = q.reshape(B, S, self.num_q_heads, self.head_dim)
        key_pre_kv = k.reshape(B, S, self.num_kv_heads, self.head_dim)
        # Repeat each KV head gqa_groups=4 times to align with num_q_heads=8.
        key_pre = mx.expand_dims(key_pre_kv, axis=-2)  # (B, S, 2, 1, 128)
        key_pre = mx.repeat(key_pre, gqa_groups, axis=-2)  # (B, S, 2, 4, 128)
        key_pre = key_pre.reshape(B, S, self.num_q_heads, self.head_dim)
        qk_mean_q = (query_pre + key_pre) / 2  # (B, S, 8, 128)
        qk_mean_k = qk_mean_q.reshape(
            B, S, self.num_kv_heads, gqa_groups, self.head_dim
        ).mean(axis=-2)  # (B, S, 2, 128)

        # ---- Two-stage causal conv ----
        # MLX nn.Conv1d expects (B, S, C); pad sequence axis on front.
        total_padding = (self.cca_time0 - 1) + (self.cca_time1 - 1)  # 2
        qk_padded = mx.pad(qk_packed0, [(0, 0), (total_padding, 0), (0, 0)])
        qk_packed3 = self.conv_qk(qk_padded)  # (B, S, 1280)

        # ---- Build queries/keys from conv output + means ----
        query = qk_packed3[..., : self.latent_q_dim].reshape(
            B, S, self.num_q_heads, self.head_dim
        ) + qk_mean_q  # (B, S, 8, 128)
        key = qk_packed3[..., self.latent_q_dim :].reshape(
            B, S, self.num_kv_heads, self.head_dim
        ) + qk_mean_k  # (B, S, 2, 128)

        # ---- Two-stream V ----
        v1 = self.val_proj1(hidden_states)  # (B, S, 128)
        # Time-shifted hidden states: drop last token, prepend a zero at front.
        # Equivalent to PyTorch's F.pad(hs[:-1], (0,0, 0,0, 1,0)) in [S,B,H] layout.
        if S > 1:
            hs_shifted = mx.pad(hidden_states[:, :-1], [(0, 0), (1, 0), (0, 0)])
        else:
            hs_shifted = mx.zeros_like(hidden_states)
        v2 = self.val_proj2(hs_shifted)  # (B, S, 128)
        value = mx.concatenate([v1, v2], axis=-1).reshape(
            B, S, self.num_kv_heads, self.head_dim
        )  # (B, S, 2, 128)

        # ---- L2 normalize Q and K, apply per-head temperature ----
        # PyTorch's `tensor.norm(p=2, dim=-1)` on a bf16 model upcasts the
        # sum-of-squares to fp32 internally for stability, then takes sqrt
        # and downcasts. Mirror that to match PyTorch's reference output.
        target_dtype = query.dtype
        q_norm = mx.sqrt(mx.sum(query.astype(mx.float32) ** 2, axis=-1, keepdims=True)).astype(target_dtype)
        k_norm = mx.sqrt(mx.sum(key.astype(mx.float32) ** 2, axis=-1, keepdims=True)).astype(target_dtype)
        # temp shape (num_kv_heads,) -> (1, 1, num_kv_heads, 1) for broadcast
        temp_b = self.temp[None, None, :, None]
        query = query * (sqrt_head_dim / q_norm)
        key = key * (sqrt_head_dim / k_norm) * temp_b

        # ---- Flatten head axis to HF flat layout ----
        query = query.reshape(B, S, self.num_q_heads * self.head_dim)
        key = key.reshape(B, S, self.num_kv_heads * self.head_dim)
        value = value.reshape(B, S, self.num_kv_heads * self.head_dim)

        return query, key, value


class ZayaAttention(nn.Module):
    """Wraps CCA + standard scaled dot product attention.

    Per modular_zaya.py:524-656. Skeleton only — forward implemented in Phase 4.
    """

    def __init__(self, args: ModelArgs, layer_number: int):
        super().__init__()
        self.qkv = CCA(args, layer_number)
        # o_proj input dim is hidden_size // 2 because CCA produces only 8
        # effective query heads (cca_num_q_heads), so the post-attention flat
        # dim is 8 * head_dim = 1024 = hidden_size // 2.
        self.o_proj = nn.Linear(
            args.hidden_size // 2,
            args.hidden_size,
            bias=args.attention_bias,
        )
        # Partial RoPE: rotates first 64 of 128 head dims; traditional=False
        # matches modular_zaya.py's apply_rotary_pos_emb (NeoX style).
        # Validated against PyTorch reference dumps within bf16 rounding noise
        # (see zaya1-mlx/validation/test_partial_rope.py).
        rotary_dim = int(
            (args.hidden_size // args.num_attention_heads) * args.partial_rotary_factor
        )
        self.rope = nn.RoPE(
            dims=rotary_dim,
            base=args.rope_theta,
            traditional=False,
        )


class ZayaRouter(nn.Module):
    """MoE router with optional EDA. Per modular_zaya.py:1036-1187.

    Skeleton only — forward implemented in Phase 6.

    EDA gate: enabled when `args.zaya_use_eda` is True AND
    layer_number != ZAYA_FIRST_MOE_LAYER (layer 1).
    """

    def __init__(self, args: ModelArgs, layer_number: int):
        super().__init__()
        self.layer_number = layer_number
        self.use_mod = args.zaya_use_mod
        self.num_experts = args.num_experts + 1 if self.use_mod else args.num_experts
        self.mlp_expansion = args.zaya_mlp_expansion

        self.down_proj = nn.Linear(args.hidden_size, self.mlp_expansion, bias=True)

        self.use_eda = args.zaya_use_eda and (layer_number != ZAYA_FIRST_MOE_LAYER)

        self.rmsnorm_eda = nn.RMSNorm(self.mlp_expansion, eps=args.norm_epsilon)
        if self.use_eda:
            self.router_states_scale = mx.ones((self.mlp_expansion,))

        # Three-layer MLP: D -> D -> D -> num_experts (with GELU between).
        # Sequential indices: 0=Linear, 1=GELU, 2=Linear, 3=GELU, 4=Linear.
        self.router_mlp = nn.Sequential(
            nn.Linear(self.mlp_expansion, self.mlp_expansion, bias=True),
            nn.GELU(),
            nn.Linear(self.mlp_expansion, self.mlp_expansion, bias=True),
            nn.GELU(),
            nn.Linear(self.mlp_expansion, self.num_experts, bias=False),
        )

        # balancing_biases is loaded from the safetensors. Init values matter
        # only as defaults if no checkpoint is loaded.
        if self.use_mod:
            init_bb = [0.0] * (self.num_experts - 1) + [-1.0]
        else:
            init_bb = [0.0] * self.num_experts
        self.balancing_biases = mx.array(init_bb)


class ZayaBlock(nn.Module):
    """MoE block: router + experts + MoD skip. Per modular_zaya.py:1329-1422.

    Skeleton only — forward implemented in Phase 6.
    """

    def __init__(self, args: ModelArgs, layer_number: int):
        super().__init__()
        self.use_mod = args.zaya_use_mod
        self.router = ZayaRouter(args, layer_number)
        # SequentialMLP holds num_experts MLPs (the skip-expert is handled by
        # passing tokens through unchanged in code; not a real MLP).
        self.experts = SequentialMLP(args, args.num_experts)


class ZayaDecoderATTLayer(nn.Module):
    """Even-indexed decoder layer (CCA self-attention).

    Per modular_zaya.py:909-1000. Skeleton only.
    """

    def __init__(self, args: ModelArgs, layer_n: int):
        super().__init__()
        self.layer_n = layer_n
        self.self_attn = ZayaAttention(args, layer_n)
        self.input_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_epsilon)
        if args.scale_residual_merge:
            self.res_scale = ResidualScaling(args, layer_n)


class ZayaDecoderMLPLayer(nn.Module):
    """Odd-indexed decoder layer (MoE).

    Per modular_zaya.py:1425-1533. Skeleton only.
    """

    def __init__(self, args: ModelArgs, layer_n: int):
        super().__init__()
        self.layer_n = layer_n
        self.zaya_block = ZayaBlock(args, layer_n)
        self.input_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_epsilon)
        if args.scale_residual_merge:
            self.res_scale = ResidualScaling(args, layer_n)


class ZayaModel(nn.Module):
    """Embedding + 80 alternating decoder layers + final ResidualScaling + final RMSNorm.

    Per modular_zaya.py:1642-1956. Skeleton only.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = []
        for layer_n in range(args.num_hidden_layers):
            if layer_n % 2 == 1:
                self.layers.append(ZayaDecoderMLPLayer(args, layer_n))
            else:
                self.layers.append(ZayaDecoderATTLayer(args, layer_n))
        if args.scale_residual_merge:
            # Final residual scaling, layer_n = num_hidden_layers (always non-first)
            self.res_scale = ResidualScaling(args, args.num_hidden_layers)
        self.final_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_epsilon)


class Model(nn.Module):
    """The mlx-lm canonical wrapper. Forward is a stub for Phase 1."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = ZayaModel(args)
        # tie_word_embeddings: lm_head is None; use embed_tokens.as_linear in __call__.
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.hidden_size, args.vocab_size, bias=args.lm_head_bias
            )

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self) -> int:
        return self.args.head_dim

    @property
    def n_kv_heads(self) -> int:
        return self.args.num_key_value_heads

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        # Phase 1 stub. Phase 9 will implement the real forward.
        raise NotImplementedError(
            "Zaya forward is implemented in Phase 9; this skeleton supports weight loading only."
        )

    def sanitize(self, weights: dict) -> dict:
        """Remap HF safetensors keys to MLX keys.

        Transformations:
          1. nn.Sequential child indexing: HF stores `conv_qk.0.weight`
             but MLX exposes `conv_qk.layers.0.weight`. Same for router_mlp.
             Insert `.layers` between the Sequential name and the index.
          2. Conv1d weight layout: PyTorch (out, in/groups, kernel) →
             MLX (out, kernel, in/groups). Transpose conv_qk weights.
          3. tie_word_embeddings: pop lm_head.weight defensively (HF
             doesn't include it for ZAYA1, but mlx-lm convention is to
             handle the general case here).
        """
        import re

        SEQ_PARENTS = re.compile(r"\.(conv_qk|router_mlp)\.(\d+)\.")

        out = {}
        for k, v in weights.items():
            new_k = SEQ_PARENTS.sub(r".\1.layers.\2.", k)
            if "self_attn.qkv.conv_qk." in new_k and new_k.endswith(".weight"):
                # PyTorch Conv1d (out, in/g, kernel) -> MLX (out, kernel, in/g)
                out[new_k] = v.transpose(0, 2, 1)
            else:
                out[new_k] = v
        if self.args.tie_word_embeddings:
            out.pop("lm_head.weight", None)
        return out
