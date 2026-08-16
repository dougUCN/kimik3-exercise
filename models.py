"""Kimi K3 skeleton for parameter counting. Build under torch.device("meta")."""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class K3Config:
    """Values from moonshotai/Kimi-K3 config.json (text_config)."""

    n_layers: int = 93
    n_dense: int = 1              # first_k_dense_replace
    d_model: int = 7168
    d_vocab: int = 163_840        # Table 1 rounds this to "160K"
    tie_embeddings: bool = False

    # Stable LatentMoE
    d_latent_moe: int = 3584      # routed_expert_hidden_size
    d_ff_expert: int = 3072       # moe_intermediate_size
    d_ff_shared: int = 3072       # shared experts run at moe_intermediate_size
    d_ff_dense: int = 33_792      # intermediate_size, used by the dense layer
    n_routed: int = 896
    n_active_routed: int = 16
    n_shared: int = 2

    # KDA
    kda_heads: int = 96
    kda_head_dim: int = 128
    kda_alpha_rank: int = 128     # rank == head dim (Kimi Linear sec 4)
    shortconv_kernel: int = 4

    # Gated MLA
    n_heads: int = 96
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    # config calls this "rope", but mla_use_nope=True and the modeling code sets
    # rotary_emb=None -- nothing is ever rotated. The name is inherited from
    # DeepSeek's MLA. Functionally these 64 dims are a head-shared (MQA) key
    # channel: computed once per token, broadcast to all heads.
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    n_mtp: int = 0                # num_nextn_predict_layers

    @property
    def n_moe_layers(self) -> int:
        return self.n_layers - self.n_dense

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


def linear(i, o, bias=False):
    return nn.Linear(i, o, bias=bias)


class ShortConv(nn.Module):
    def __init__(self, dim, kernel):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim, bias=False)


class KDA(nn.Module):
    """Kimi Delta Attention. Eq. 2 + Eq. 6 of the K3 paper."""

    def __init__(self, c: K3Config):
        super().__init__()
        h, dk = c.kda_heads, c.kda_head_dim
        inner = h * dk

        self.q_proj = linear(c.d_model, inner)
        self.k_proj = linear(c.d_model, inner)
        self.v_proj = linear(c.d_model, inner)
        self.q_conv = ShortConv(inner, c.shortconv_kernel)
        self.k_conv = ShortConv(inner, c.shortconv_kernel)
        self.v_conv = ShortConv(inner, c.shortconv_kernel)

        self.beta_proj = linear(c.d_model, h)
        self.alpha_down = linear(c.d_model, c.kda_alpha_rank)
        self.alpha_up = linear(c.kda_alpha_rank, inner)
        self.alpha_bias = nn.Parameter(torch.empty(h, dk))
        self.log_scale = nn.Parameter(torch.empty(h))

        self.g_proj = linear(c.d_model, inner)          # full-rank gate
        self.o_proj = linear(inner, c.d_model)
        self.o_norm = nn.RMSNorm(dk)


class GatedMLA(nn.Module):
    """Multi-head Latent Attention, NoPE, with full-rank output gate (Eq. 7).

    No positional encoding at all -- the KDA layers carry position. The 64
    "rope" dims are real parameters but unrotated; see K3Config.
    """

    def __init__(self, c: K3Config):
        super().__init__()
        n, v = c.n_heads, c.v_head_dim

        self.q_down = linear(c.d_model, c.q_lora_rank)
        self.q_up = linear(c.q_lora_rank, n * c.qk_head_dim)
        self.kv_down = linear(c.d_model, c.kv_lora_rank + c.qk_rope_head_dim)
        self.kv_up = linear(c.kv_lora_rank, n * (c.qk_nope_head_dim + v))
        self.q_norm = nn.RMSNorm(c.q_lora_rank)
        self.kv_norm = nn.RMSNorm(c.kv_lora_rank)

        self.g_proj = linear(c.d_model, n * v)          # full-rank gate
        self.o_proj = linear(n * v, c.d_model)


class GLU(nn.Module):
    """SiTU-GLU has the same parameter shapes as SwiGLU (Eq. 12)."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = linear(dim, hidden)
        self.up = linear(dim, hidden)
        self.down = linear(hidden, dim)


class StableLatentMoE(nn.Module):
    """Eq. 11. Routed experts live in the latent space of width d_latent_moe."""

    def __init__(self, c: K3Config):
        super().__init__()
        self.c = c
        self.router = linear(c.d_model, c.n_routed)
        self.router_bias = nn.Parameter(torch.empty(c.n_routed))

        self.shared = nn.ModuleList(
            GLU(c.d_model, c.d_ff_shared) for _ in range(c.n_shared)
        )
        self.down = linear(c.d_model, c.d_latent_moe)
        self.experts = nn.ModuleList(
            GLU(c.d_latent_moe, c.d_ff_expert) for _ in range(c.n_routed)
        )
        self.u_norm = nn.RMSNorm(c.d_latent_moe)
        self.up = linear(c.d_latent_moe, c.d_model)

    def active_params(self) -> int:
        """Params touched by one token: shared + down/up + k routed experts."""
        per_expert = sum(p.numel() for p in self.experts[0].parameters())
        rest = sum(p.numel() for p in self.parameters()) - sum(
            p.numel() for p in self.experts.parameters()
        )
        return rest + self.c.n_active_routed * per_expert


class Block(nn.Module):
    def __init__(self, c: K3Config, attn: str, dense: bool):
        super().__init__()
        self.attn = KDA(c) if attn == "kda" else GatedMLA(c)
        self.ffn = (
            GLU(c.d_model, c.d_ff_dense) if dense else StableLatentMoE(c)
        )
        self.attn_norm = nn.RMSNorm(c.d_model)
        self.ffn_norm = nn.RMSNorm(c.d_model)
        self.attn_res_q = nn.Parameter(torch.empty(c.d_model))  # AttnRes, Eq. 8

    def active_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        if isinstance(self.ffn, StableLatentMoE):
            n -= sum(p.numel() for p in self.ffn.parameters())
            n += self.ffn.active_params()
        return n


def layer_plan(c: K3Config):
    """3 KDA + 1 MLA per block, plus a trailing MLA layer (sec 2.1).

    Reproduces linear_attn_config.full_attn_layers, which is 1-indexed and
    ends [..., 88, 92, 93] -- layer 93 is the extra global-attention layer.
    """
    return [
        "mla" if (i % 4 == 0 or i == c.n_layers) else "kda"
        for i in range(1, c.n_layers + 1)
    ]


class KimiK3(nn.Module):
    def __init__(self, c: K3Config = K3Config()):
        super().__init__()
        self.c = c
        plan = layer_plan(c)
        self.embed = nn.Embedding(c.d_vocab, c.d_model)
        self.blocks = nn.ModuleList(
            Block(c, attn, dense=(i < c.n_dense)) for i, attn in enumerate(plan)
        )
        self.norm = nn.RMSNorm(c.d_model)
        self.head = None if c.tie_embeddings else linear(c.d_model, c.d_vocab)
        self.mtp = nn.ModuleList(
            Block(c, "mla", dense=False) for _ in range(c.n_mtp)
        )

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def active_params(self, count_embed=False) -> int:
        """Params touched per token. The embedding is a lookup, not a matmul,
        so it is excluded by default; the output head is counted."""
        n = sum(b.active_params() for b in self.blocks)
        n += self.norm.weight.numel()
        if self.head is not None:
            n += self.head.weight.numel()
        if count_embed:
            n += self.embed.weight.numel()
        return n

    def breakdown(self, active=False) -> dict:
        """Params by component. active=True counts only what fires per token:
        16 of 896 routed experts, and no embedding lookup."""

        def n(mod):
            return sum(p.numel() for p in mod.parameters())

        out = {}
        for name, pick in [
            ("KDA", lambda b: isinstance(b.attn, KDA)),
            ("Gated MLA", lambda b: isinstance(b.attn, GatedMLA)),
        ]:
            out[name] = sum(n(b.attn) for b in self.blocks if pick(b))

        moe = [b.ffn for b in self.blocks if isinstance(b.ffn, StableLatentMoE)]
        out["MoE routed"] = sum(n(f.experts) for f in moe)
        out["MoE shared"] = sum(n(f.shared) for f in moe)
        out["MoE down/up"] = sum(n(f.down) + n(f.up) for f in moe)
        out["MoE router"] = sum(
            n(f.router) + f.router_bias.numel() + f.u_norm.weight.numel() for f in moe
        )
        out["Dense FFN"] = sum(
            n(b.ffn) for b in self.blocks if not isinstance(b.ffn, StableLatentMoE)
        )
        out["Embedding"] = n(self.embed)
        out["Output head"] = 0 if self.head is None else n(self.head)
        out["Other"] = self.total_params() - sum(out.values())

        if active:
            out["MoE routed"] = (
                out["MoE routed"] * self.c.n_active_routed // self.c.n_routed
            )
            out["Embedding"] = 0
        return out


def build(c: K3Config = K3Config()) -> KimiK3:
    with torch.device("meta"):
        return KimiK3(c)
