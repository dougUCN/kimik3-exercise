"""Per-operation roofline for Kimi K3 inference.

Every op gets a FLOP count and a byte count. Divide each by the machine's peak
and you get two times; the larger one is what you actually wait for.

    t_compute = flops / peak_flops(precision)
    t_memory  = bytes / bandwidth

Serving precision follows the paper (sec 4.1.4): MoE expert weights in MXFP4
with MXFP8 activations, everything else left in higher precision (BF16).
"""

from dataclasses import dataclass, field

import pandas as pd

# ---------------------------------------------------------------- precision

# MXFP4 is 4-bit elements plus one FP8 scale per block of 32 -> 4 + 8/32 bits.
BYTES = {"bf16": 2.0, "fp8": 1.0, "mxfp4": (4 + 8 / 32) / 8}


@dataclass
class GPU:
    """NVIDIA GB300 NVL72 bin, per Blackwell Ultra (B300) GPU.

    All figures DENSE (no 2:4 sparsity). Sources:
      288 GB HBM3e, 8 TB/s, NVFP4 15 PF dense, FP8 5 PF dense
        -- developer.nvidia.com "Inside NVIDIA Blackwell Ultra" (per-GPU table)
      BF16 2.5 PF dense -- derived: rack 360 PF sparse / 2 / 72 GPUs.
        NVIDIA publishes no per-GPU BF16 figure for Blackwell Ultra.

    Blackwell Ultra raised NVFP4 1.5x (10 -> 15 PF dense) and left FP8, BF16 and
    TF32 exactly at B200 rates. Do NOT derive FP4 dense by halving a published
    sparse number: NVIDIA lists 15 dense | 20 sparse (1.33x, not 2x), so halving
    20 gives B200's 10 PF, not B300's. That is the easiest error to make here.

    MXFP4 runs the same tensor-core datapath as NVFP4 at the same peak; NVIDIA
    does not publish an MXFP4 figure separately.
    """

    name: str = "GB300 NVL72 (per B300 GPU)"
    hbm_bytes: float = 288 * 1024**3  # 288 GB HBM3e
    bandwidth: float = 8.0e12         # bytes/s
    peak: dict = field(
        default_factory=lambda: {
            "bf16": 2.5e15,
            "fp8": 5.0e15,
            "mxfp4": 15.0e15,
        }
    )

    @property
    def balance(self):
        """FLOPs per byte the machine can sustain. Above this an op is
        compute bound, below it the op is waiting on memory."""
        return {k: v / self.bandwidth for k, v in self.peak.items()}


def hgx_b300():
    """The air-cooled HGX/DGX B300 bin -- about 10% below the NVL72 part."""
    return GPU(
        name="HGX/DGX B300 (per GPU)",
        hbm_bytes=288 * 1024**3,
        bandwidth=8.0e12,
        peak={"bf16": 2.25e15, "fp8": 4.5e15, "mxfp4": 13.5e15},
    )


# ---------------------------------------------------------------- operations


def _gemm(name, group, count, d_in, d_out, batch, prec="bf16", n_weights=1):
    """A dense matmul: batch tokens through an (d_in x d_out) weight."""
    params = n_weights * d_in * d_out
    return dict(
        op=name,
        group=group,
        count=count,
        precision=prec,
        flops=count * 2 * batch * params,
        weight_bytes=count * params * BYTES[prec],
        act_bytes=count * batch * (d_in + n_weights * d_out) * BYTES["bf16"],
    )


def experts_touched(n_routed, k, batch):
    """Distinct experts at least one token in the batch selects.

    One token touches k. A big batch touches nearly all of them -- which is
    why MoE decode gets *more* memory hungry as you batch, not less.
    """
    return n_routed * (1 - (1 - k / n_routed) ** batch)


def build_ops(c, batch=1, seq=131_072, hw=None):
    """One row per operation for a single decode step."""
    hw = hw or GPU()
    d = c.d_model
    inner = c.kda_heads * c.kda_head_dim
    plan = ["mla" if (i % 4 == 0 or i == c.n_layers) else "kda" for i in range(1, c.n_layers + 1)]
    n_mla, n_kda = plan.count("mla"), plan.count("kda")
    n_moe = c.n_moe_layers

    ops = [
        # ---- KDA ----
        _gemm("kda.qkv_proj", "KDA", n_kda, d, inner, batch, n_weights=3),
        _gemm("kda.gate", "KDA", n_kda, d, inner, batch),
        _gemm("kda.out", "KDA", n_kda, inner, d, batch),
        _gemm("kda.alpha", "KDA", n_kda, d, c.kda_alpha_rank, batch),
        _gemm("kda.alpha_up", "KDA", n_kda, c.kda_alpha_rank, inner, batch),
        _gemm("kda.beta", "KDA", n_kda, d, c.kda_heads, batch),
        # ---- MLA ----
        _gemm("mla.q_down", "MLA", n_mla, d, c.q_lora_rank, batch),
        _gemm("mla.q_up", "MLA", n_mla, c.q_lora_rank, c.n_heads * c.qk_head_dim, batch),
        _gemm("mla.kv_down", "MLA", n_mla, d, c.kv_lora_rank + c.qk_rope_head_dim, batch),
        _gemm("mla.gate", "MLA", n_mla, d, c.n_heads * c.v_head_dim, batch),
        _gemm("mla.out", "MLA", n_mla, c.n_heads * c.v_head_dim, d, batch),
        # ---- MoE ----
        _gemm("moe.router", "MoE", n_moe, d, c.n_routed, batch),
        _gemm("moe.w_down", "MoE", n_moe, d, c.d_latent_moe, batch),
        _gemm("moe.w_up", "MoE", n_moe, c.d_latent_moe, d, batch),
        _gemm("moe.shared", "MoE", n_moe, d, c.d_ff_shared, batch, n_weights=3 * c.n_shared),
        # ---- head ----
        _gemm("lm_head", "Head", 1, d, c.d_vocab, batch),
    ]

    # ---- routed experts: weights read scale with DISTINCT experts, flops with k
    live = experts_touched(c.n_routed, c.n_active_routed, batch)
    per_expert = 3 * c.d_latent_moe * c.d_ff_expert
    ops.append(
        dict(
            op="moe.experts",
            group="MoE",
            count=n_moe,
            precision="mxfp4",
            flops=n_moe * 2 * batch * c.n_active_routed * per_expert,
            weight_bytes=n_moe * live * per_expert * BYTES["mxfp4"],
            act_bytes=n_moe * batch * c.n_active_routed
            * (c.d_latent_moe + 3 * c.d_ff_expert) * BYTES["fp8"],
        )
    )

    # ---- MLA attention against the KV cache: no weights, reads the whole cache
    cache = seq * (c.kv_lora_rank + c.qk_rope_head_dim) * BYTES["bf16"]
    ops.append(
        dict(
            op="mla.attn (vs cache)",
            group="MLA",
            count=n_mla,
            precision="bf16",
            flops=n_mla * 2 * batch * seq * c.n_heads * (c.qk_head_dim + c.v_head_dim),
            weight_bytes=0.0,
            act_bytes=n_mla * batch * cache,
        )
    )

    # ---- KDA recurrence: no weights, reads and writes a fixed-size state
    state = c.kda_heads * c.kda_head_dim * c.kda_head_dim * BYTES["bf16"]
    ops.append(
        dict(
            op="kda.recurrence",
            group="KDA",
            count=n_kda,
            precision="bf16",
            flops=n_kda * batch * 6 * c.kda_heads * c.kda_head_dim * c.kda_head_dim,
            weight_bytes=0.0,
            act_bytes=n_kda * batch * 2 * state,  # read + write
        )
    )

    df = pd.DataFrame(ops)
    df["bytes"] = df.weight_bytes + df.act_bytes
    df["intensity"] = df.flops / df.bytes
    # what the compute bound would be at each precision -- the op's own
    # precision picks the one that actually applies
    for prec, peak in hw.peak.items():
        df[f"t_compute_{prec}"] = df.flops / peak
    df["t_compute"] = df.flops / df.precision.map(hw.peak)
    df["t_memory"] = df.bytes / hw.bandwidth
    df["bound"] = ["compute" if c_ > m else "memory" for c_, m in zip(df.t_compute, df.t_memory)]
    df["t"] = df[["t_compute", "t_memory"]].max(axis=1)
    return df.set_index("op").sort_values("t", ascending=False)


def summarise(df, hw=None):
    hw = hw or GPU()
    tot = df[["flops", "bytes", "t_compute", "t_memory", "t"]].sum()
    return pd.Series(
        {
            "TFLOP / token": tot.flops / 1e12,
            "GB moved / token": tot.bytes / 1e9,
            "arithmetic intensity": tot.flops / tot.bytes,
            "machine balance (bf16)": hw.balance["bf16"],
            "t_compute (ms)": tot.t_compute * 1e3,
            "t_memory (ms)": tot.t_memory * 1e3,
            "t_total (ms)": tot.t * 1e3,
            "tokens/s (1 GPU)": 1 / tot.t,
        }
    )


def by_layer(df):
    """Runtime for ONE layer of each type, and the model total.

    Per op we take max(t_compute, t_memory) -- the resource that op waits on --
    then sum those maxima within a layer. That assumes ops run back to back and
    each one saturates a single resource. Real kernels overlap and fuse, so
    treat this as a ceiling on how well you could do, not a prediction.
    """
    d = df.copy()
    d["t_per_instance"] = d.t / d["count"]

    out = d.groupby("group").apply(
        lambda g: pd.Series(
            {
                "instances": int(g["count"].max()),
                "t_1layer_us": g.t_per_instance.sum() * 1e6,
                "t_total_ms": g.t.sum() * 1e3,
                "compute-bound ops": int((g.bound == "compute").sum()),
                "memory-bound ops": int((g.bound == "memory").sum()),
            }
        ),
        include_groups=False,
    )
    out = out.sort_values("t_total_ms", ascending=False)
    out.loc["MODEL"] = [
        out.instances.sum(),
        float("nan"),  # summing per-layer times across types is meaningless
        out.t_total_ms.sum(),
        int((d.bound == "compute").sum()),
        int((d.bound == "memory").sum()),
    ]
    return out


def footprint(c, hw=None):
    """Does it fit? Weights at serving precision vs one GPU's HBM."""
    hw = hw or GPU()
    n_moe = c.n_moe_layers
    experts = n_moe * c.n_routed * 3 * c.d_latent_moe * c.d_ff_expert
    from models import build

    total = build(c).total_params()
    rest = total - experts
    b = experts * BYTES["mxfp4"] + rest * BYTES["bf16"]
    return pd.Series(
        {
            "expert params (B)": experts / 1e9,
            "other params (B)": rest / 1e9,
            "weights MXFP4+BF16 (GB)": b / 1e9,
            "weights all BF16 (GB)": total * BYTES["bf16"] / 1e9,
            "one GPU HBM (GB)": hw.hbm_bytes / 1e9,
            "GPUs to hold weights": b / hw.hbm_bytes,
        }
    )


def op_table(df, hw=None):
    """The one table: work, traffic, intensity, and both bounds per operation.

    COMPUTE BOUND columns are t = flops / peak_flops(precision), i.e. how long
    the math alone would take. Three of them, so you can see what quantising
    further would buy. Bytes are held fixed across the three -- this answers
    "if the arithmetic ran faster, would it help?", not "what if we requantised
    the weights", which would move the traffic too.

    MEMORY BOUND is t = bytes / bandwidth: how long the DRAM traffic alone takes.

    An operation waits on whichever is larger, so `limited by` is just the
    argmax. Equivalently, compare `intensity` against the machine balance
    (peak_flops / bandwidth): above it compute bound, below it memory bound.
    """
    hw = hw or GPU()
    out = pd.DataFrame(
        {
            "GFLOP": df.flops / 1e9,
            "GB from DRAM": df.bytes / 1e9,
            "intensity (FLOP/byte)": df.intensity,
            "COMPUTE bf16 (us)": df.t_compute_bf16 * 1e6,
            "COMPUTE fp8 (us)": df.t_compute_fp8 * 1e6,
            "COMPUTE mxfp4 (us)": df.t_compute_mxfp4 * 1e6,
            "MEMORY (us)": df.t_memory * 1e6,
        },
        index=df.index,
    )
    out.insert(0, "group", df.group)
    out["limited by"] = df.bound
    return out


def balance_table(hw=None):
    """Machine balance: FLOPs per byte the GPU can sustain at each precision.

    An op needs at least this much arithmetic per byte of DRAM traffic to keep
    the tensor cores busy. Below it, you are waiting on memory no matter what.
    """
    hw = hw or GPU()
    return pd.Series(
        {f"{k} (FLOP/byte)": v for k, v in hw.balance.items()}, name=hw.name
    )


def machine_intensity(hw=None):
    """The GPU's own arithmetic intensity, a.k.a. machine balance or the
    ridge point: peak FLOP/s divided by bandwidth, in FLOP/byte.

    This is how much arithmetic the hardware can do per byte it can fetch.
    An op must supply at least this much work per byte to keep the tensor
    cores fed; supply less and you are bandwidth limited no matter what.
    """
    hw = hw or GPU()
    return pd.DataFrame(
        {
            "peak (PFLOP/s)": {k: v / 1e15 for k, v in hw.peak.items()},
            "bandwidth (TB/s)": {k: hw.bandwidth / 1e12 for k in hw.peak},
            "machine intensity (FLOP/byte)": hw.balance,
        }
    )


def vs_ridge(df, hw=None):
    """How far each op sits below the machine's ridge point.

    ratio = op intensity / machine intensity, at the op's own precision.
    Below 1.0 means memory bound, by that factor.
    """
    hw = hw or GPU()
    ridge = df.precision.map(hw.balance)
    out = pd.DataFrame(
        {
            "group": df.group,
            "precision": df.precision,
            "op intensity": df.intensity,
            "GB300 intensity": ridge,
            "ratio": df.intensity / ridge,
        },
        index=df.index,
    )
    out["shortfall"] = [f"{1 / r:,.0f}x under" if r < 1 else f"{r:.1f}x over" for r in out.ratio]
    return out


def explain_gemm(name, d_in, d_out, layers, batch=1, prec="bf16", n_weights=1, heads=None):
    """Print the FLOP and byte arithmetic for one GEMM, term by term."""
    bpw = BYTES[prec]
    params = n_weights * d_in * d_out
    flops = layers * 2 * batch * params
    wbytes = layers * params * bpw
    abytes = layers * batch * (d_in + n_weights * d_out) * BYTES["bf16"]

    print(f"{name}   ({layers} layers, batch {batch}, {prec})")
    if heads:
        print(f"  d_out  = {heads[0]} heads x {heads[1]} = {d_out}"
              f"   <- heads are folded into one width, not a repeated matmul")
    print(f"  params/layer  {n_weights} x {d_in} x {d_out} = {params:,}")
    print()
    print(f"  FLOPs   {layers} x 2 x {batch} x {params:,}")
    print(f"          = {flops:,} = {flops / 1e9:.2f} GFLOP")
    print(f"            the 2 is one multiply + one add per parameter")
    print()
    print(f"  weights {layers} x {params:,} x {bpw} B = {wbytes / 1e9:.2f} GB")
    print(f"  activs  {layers} x {batch} x ({d_in} + {n_weights}x{d_out}) x 2 B = {abytes / 1e9:.4f} GB")
    print(f"          = {(wbytes + abytes) / 1e9:.2f} GB total")
    print()
    print(f"  intensity = {flops / 1e9:.2f} / {(wbytes + abytes) / 1e9:.2f} = {flops / (wbytes + abytes):.5f} FLOP/byte")
    print(f"  ideal (weights only) = 2 x {batch} / {bpw} = {2 * batch / bpw:.3f}")
    return flops / (wbytes + abytes)


def build_ops_prefill(c, chunk=4096, prior=0, hw=None):
    """One row per operation for a PREFILL chunk.

    Real engines chunk prefill rather than swallowing a whole prompt, so this
    models one chunk of `chunk` tokens with `prior` tokens already cached.

    Three things differ from decode:
      * every GEMM runs `chunk` tokens through the same weights, so FLOPs scale
        with chunk while weight bytes do not -- this is what lifts intensity
      * attention is causal over prior+chunk keys, so its FLOPs go quadratic
      * with thousands of tokens routing independently, essentially every expert
        wakes, so the whole bank is read rather than 16 per layer
    """
    hw = hw or GPU()
    d = c.d_model
    inner = c.kda_heads * c.kda_head_dim
    plan = ["mla" if (i % 4 == 0 or i == c.n_layers) else "kda" for i in range(1, c.n_layers + 1)]
    n_mla, n_kda = plan.count("mla"), plan.count("kda")
    n_moe = c.n_moe_layers
    B = chunk

    ops = [
        _gemm("kda.qkv_proj", "KDA", n_kda, d, inner, B, n_weights=3),
        _gemm("kda.gate", "KDA", n_kda, d, inner, B),
        _gemm("kda.out", "KDA", n_kda, inner, d, B),
        _gemm("kda.alpha", "KDA", n_kda, d, c.kda_alpha_rank, B),
        _gemm("kda.alpha_up", "KDA", n_kda, c.kda_alpha_rank, inner, B),
        _gemm("kda.beta", "KDA", n_kda, d, c.kda_heads, B),
        _gemm("mla.q_down", "MLA", n_mla, d, c.q_lora_rank, B),
        _gemm("mla.q_up", "MLA", n_mla, c.q_lora_rank, c.n_heads * c.qk_head_dim, B),
        _gemm("mla.kv_down", "MLA", n_mla, d, c.kv_lora_rank + c.qk_rope_head_dim, B),
        _gemm("mla.gate", "MLA", n_mla, d, c.n_heads * c.v_head_dim, B),
        _gemm("mla.out", "MLA", n_mla, c.n_heads * c.v_head_dim, d, B),
        _gemm("moe.router", "MoE", n_moe, d, c.n_routed, B),
        _gemm("moe.w_down", "MoE", n_moe, d, c.d_latent_moe, B),
        _gemm("moe.w_up", "MoE", n_moe, c.d_latent_moe, d, B),
        _gemm("moe.shared", "MoE", n_moe, d, c.d_ff_shared, B, n_weights=3 * c.n_shared),
        _gemm("lm_head", "Head", 1, d, c.d_vocab, B),
    ]

    live = experts_touched(c.n_routed, c.n_active_routed, B)
    per_expert = 3 * c.d_latent_moe * c.d_ff_expert
    ops.append(
        dict(
            op="moe.experts",
            group="MoE",
            count=n_moe,
            precision="mxfp4",
            flops=n_moe * 2 * B * c.n_active_routed * per_expert,
            weight_bytes=n_moe * live * per_expert * BYTES["mxfp4"],
            act_bytes=n_moe * B * c.n_active_routed
            * (c.d_latent_moe + 3 * c.d_ff_expert) * BYTES["fp8"],
        )
    )

    # causal pairs: each query in the chunk sees all `prior` keys plus the part
    # of its own chunk up to itself -> chunk*prior + chunk*(chunk+1)/2
    pairs = B * prior + B * (B + 1) / 2
    entry = (c.kv_lora_rank + c.qk_rope_head_dim) * BYTES["bf16"]
    ops.append(
        dict(
            op="mla.attn (causal)",
            group="MLA",
            count=n_mla,
            precision="bf16",
            flops=n_mla * 2 * pairs * c.n_heads * (c.qk_head_dim + c.v_head_dim),
            weight_bytes=0.0,
            act_bytes=n_mla * (prior + B) * entry,  # read prior, write this chunk
        )
    )

    # KDA chunkwise scan: state touched once per chunk, work scales with tokens
    state = c.kda_heads * c.kda_head_dim * c.kda_head_dim * BYTES["bf16"]
    ops.append(
        dict(
            op="kda.recurrence",
            group="KDA",
            count=n_kda,
            precision="bf16",
            flops=n_kda * B * 6 * c.kda_heads * c.kda_head_dim * c.kda_head_dim,
            weight_bytes=0.0,
            act_bytes=n_kda * (2 * state + B * inner * BYTES["bf16"]),
        )
    )

    df = pd.DataFrame(ops)
    df["bytes"] = df.weight_bytes + df.act_bytes
    df["intensity"] = df.flops / df.bytes
    for prec, peak in hw.peak.items():
        df[f"t_compute_{prec}"] = df.flops / peak
    df["t_compute"] = df.flops / df.precision.map(hw.peak)
    df["t_memory"] = df.bytes / hw.bandwidth
    df["bound"] = ["compute" if c_ > m else "memory" for c_, m in zip(df.t_compute, df.t_memory)]
    df["t"] = df[["t_compute", "t_memory"]].max(axis=1)
    return df.set_index("op").sort_values("t", ascending=False)
