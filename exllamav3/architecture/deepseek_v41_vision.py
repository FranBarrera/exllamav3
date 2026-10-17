from __future__ import annotations

import torch
import torch.nn.functional as F
from typing_extensions import override

from ..model.model import Model
from ..modules import GatedMLP, Linear, RMSNorm
from ..modules.module import Module

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .deepseek_v41 import DeepseekV41Config


# Learned boundary rows copied verbatim. There is no image_pad tensor in V4.1.
ORPHAN_TENSORS = ("image_start", "image_end", "image_newline")
_COPY_PREFIXES = ("vision", "aligner", "image_start", "image_end", "image_newline")


def vision_rope(n_h: int, n_w: int, dim: int, theta: float, device, dtype):
    """2D rotary cos/sin over head_dim/2, one pair axis for height and one for width."""
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype = torch.float32, device = device) / dim))
    hpos = torch.arange(n_h, device = device).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w, device = device).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim = -1).reshape(-1, 2, 1).float() * inv
    freqs = freqs.flatten(1)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _rotate_half_pair(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.float().chunk(2, dim = -1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim = -1).to(dtype = x.dtype)


class DSV41VisionBlock(Module):
    def __init__(self, config, idx: int, vision):
        key = f"vision.blocks.{idx}"
        super().__init__(config, key, None)
        self.layer_idx = idx
        self.n_heads = vision.n_heads
        self.head_dim = vision.head_dim
        self.norm1 = RMSNorm(config, f"{key}.norm1", vision.rms_norm_eps)
        self.wqkv = Linear(
            config, f"{key}.attn.wqkv", vision.dim, 3 * vision.dim,
            qmap = "block.attn", out_dtype = torch.float, pad_to = 1,
        )
        self.wo = Linear(
            config, f"{key}.attn.wo", vision.dim, vision.dim,
            qmap = "block.attn", out_dtype = torch.float, pad_to = 1,
        )
        self.norm2 = RMSNorm(config, f"{key}.norm2", vision.rms_norm_eps)
        self.mlp = GatedMLP(
            config = config,
            key = f"{key}.mlp",
            hidden_size = vision.dim,
            intermediate_size = vision.inter_dim,
            key_fused_gate_up = "w1",
            key_gate = "gate_proj",
            key_up = "up_proj",
            key_down = "w2",
            activation_fn = "silu",
            qmap = "block.mlp",
            pad_to = 1,
        )
        self.extra_source_keys = (f"{key}.attn.wqkv.bias", f"{key}.attn.wo.bias")
        for module in (self.norm1, self.wqkv, self.wo, self.norm2, self.mlp):
            self.register_submodule(module)

    @override
    def optimizer_targets(self):
        return [self.wqkv.key, self.wo.key, self.mlp.optimizer_targets()]

    @override
    def forward(self, x, params, out_dtype = None):
        cos, sin = params["dsv41_vision_rope"]
        h = self.norm1.forward(x, params)
        qkv = self.wqkv.forward(h, params)
        q, k, v = (t.view(-1, self.n_heads, self.head_dim) for t in qkv.chunk(3, dim = -1))
        q = _rotate_half_pair(q, cos, sin)
        k = _rotate_half_pair(k, cos, sin)
        out = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        out = self.wo.forward(out.transpose(0, 1).reshape(x.shape[0], -1), params)
        x = x + out
        y = self.mlp.forward(self.norm2.forward(x, params), params)
        return x + y


class DSV41VisionAligner(Module):
    def __init__(self, config, vision, out_hidden: int):
        super().__init__(config, "aligner", None)
        r = vision.downsample_ratio
        self.downsample_ratio = r
        in_dim = vision.dim * r * r
        self.w1 = Linear(
            config, "aligner.w1", in_dim, out_hidden,
            qmap = "block", out_dtype = torch.float, pad_to = 1,
        )
        self.w2 = Linear(
            config, "aligner.w2", out_hidden, out_hidden,
            qmap = "block", out_dtype = torch.float, pad_to = 1,
        )
        self.extra_source_keys = ("aligner.w1.bias", "aligner.w2.bias")
        self.register_submodule(self.w1)
        self.register_submodule(self.w2)

    @override
    def optimizer_targets(self):
        return [self.w1.key, self.w2.key]

    @override
    def forward(self, x, params, out_dtype = None):
        n_h, n_w = params["dsv41_vision_grid"]
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride = r).squeeze(0).transpose(0, 1)
        return self.w2.forward(F.gelu(self.w1.forward(x, params)), params)


class DeepseekV41VisionModel(Model):
    """ViT + gelu aligner.

    The default ``vision_bits`` stays 16, so convert copies the tower raw (the checkpoint
    stores these weights without ``.scale``). Passing ``--vision_bits`` 1-8 quantizes the
    Linears. The generator does not yet feed images through this tower.
    """

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: DeepseekV41Config) -> dict:
        found = {}
        for prefix in _COPY_PREFIXES:
            found.update(config.stc.list_tensors(prefix = prefix))
        return found

    def __init__(self, config: DeepseekV41Config, **kwargs):
        super().__init__(config, **kwargs)
        vision = config.vision
        if vision is None:
            raise RuntimeError("DeepseekV41 vision component requires vision_config")
        # Leave default_vision_bits unset so convert keeps the tower unquantized unless asked.
        self.caps.update({"supports_tp": False})
        patch = Linear(
            config = config,
            key = "vision.patch_embed.proj",
            in_features = vision.num_channels * vision.patch_size ** 2,
            out_features = vision.dim,
            qmap = "block.attn",
            out_dtype = torch.float,
            pad_to = 1,
        )
        patch.extra_source_keys = ("vision.patch_embed.proj.bias",)
        self.modules.append(patch)
        self.first_block_idx = len(self.modules)
        for idx in range(vision.n_layers):
            self.modules.append(DSV41VisionBlock(config, idx, vision))
        self.last_kv_module_idx = len(self.modules) - 1
        self.modules += [
            RMSNorm(config, "vision.norm", vision.rms_norm_eps, out_dtype = torch.half),
            DSV41VisionAligner(config, vision, config.hidden_size),
        ]

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        raise NotImplementedError(
            "DeepSeek-V4.1 vision is converted with the text model, but image prefill is not wired"
        )

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        return prompt

    def forward_patches(self, patches: torch.Tensor, n_h: int, n_w: int, params: dict | None = None):
        """patches [n_h * n_w, 3 * patch * patch] -> aligner rows. Not used by the generator."""
        params = dict(params or {})
        vision = self.config.vision
        cos, sin = vision_rope(
            n_h, n_w, vision.head_dim // 2, vision.rope_theta, patches.device, patches.dtype,
        )
        params["dsv41_vision_rope"] = (cos, sin)
        params["dsv41_vision_grid"] = (n_h, n_w)
        x = patches
        for module in self.modules:
            x = module.forward(x, params)
        return x
