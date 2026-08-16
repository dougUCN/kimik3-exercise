"""Display helpers for walking through matmuls and their FLOP cost."""

import numpy as np


def show(*parts, prec=0):
    """Print matrices side by side: show(("A", A), "@", ("B", B), "=", ("C", C))"""
    cols = []
    for p in parts:
        if isinstance(p, str):
            cols.append(("", [p]))
        else:
            name, M = p
            M = np.atleast_2d(M)
            rows = [" ".join(f"{v:{prec + 5}.{prec}f}" for v in r) for r in M]
            cols.append((f"{name} ({M.shape[0]}x{M.shape[1]})", rows))

    w = [max(len(lbl), max(len(r) for r in rows)) for lbl, rows in cols]
    h = max(len(rows) for _, rows in cols)

    print("   ".join(lbl.center(wi) for (lbl, _), wi in zip(cols, w)))
    for i in range(h):
        line = []
        for (_, rows), wi in zip(cols, w):
            top = (h - len(rows)) // 2
            j = i - top
            line.append((rows[j] if 0 <= j < len(rows) else "").center(wi))
        print("   ".join(line))


def matmul_counted(A, B):
    """A @ B, tallying every scalar multiply and add.

    Returns (C, mults, adds). Each output cell is a dot product of length k:
    k multiplies and k-1 adds, since the first term initialises the accumulator.
    """
    m, k = A.shape
    _, n = B.shape
    C = np.zeros((m, n))
    mults = adds = 0
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for p in range(k):
                acc += A[i, p] * B[p, j]
                mults += 1
                adds += 1
            adds -= 1
            C[i, j] = acc
    return C, mults, adds


def matmul_step(a, b, out_name, prec=0, note=None):
    """Show `A @ B = C` with its shapes and FLOP accounting. Returns C."""
    (a_name, A), (b_name, B) = a, b
    C, mults, adds = matmul_counted(A, B)
    m, k = A.shape
    _, n = B.shape

    show((a_name, A), "@", (b_name, B), "=", (out_name, C), prec=prec)
    print(f"\n  shapes      ({m}x{k}) @ ({k}x{n}) -> ({m}x{n})")
    print(f"  counted     {mults} mults + {adds} adds = {mults + adds}")
    print(f"  2*m*k*n     2*{m}*{k}*{n} = {2 * m * k * n}   (rounds up by m*n = {m * n})")
    if note:
        print(f"  note        {note}")
    return C


def softmax_step(scores, d_k, causal=True, prec=2):
    """Show scores -> softmax -> attention weights. Returns the weights."""
    s = scores.shape[0]
    x = scores / np.sqrt(d_k)
    if causal:
        x = x + np.triu(np.full((s, s), -1e9), 1)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    attn = e / e.sum(axis=1, keepdims=True)

    show(("scores", scores), "-> softmax ->", ("attn weights", attn), prec=prec)
    print(f"\n  rows sum to 1   {attn.sum(axis=1)}")
    if causal:
        print("  row i sees only columns <= i -- zeros above the diagonal are the mask")
    print(f"  FLOPs           ~{s * s} elementwise, not a matmul -- small beside the two that are")
    return attn


def kda_recurrence(Q, K, V, alpha, beta, verbose=True):
    """KDA / delta-rule recurrence, Eq. 1 of the K3 paper.

        S_t = (I - b_t k_t k_t^T) diag(a_t) S_{t-1} + b_t k_t v_t^T
        o_t = S_t^T q_t

    S is (d_k, d_v) and never changes shape -- that is what makes KDA linear.
    L2Norm / ShortConv / Swish on q,k,v are omitted here for clarity.
    Returns (outputs, final_state).
    """
    s, d_k = Q.shape
    d_v = V.shape[1]
    S = np.zeros((d_k, d_v))
    outs = []

    for t in range(s):
        q = Q[t][:, None]
        k = K[t][:, None]
        v = V[t][None, :]
        a = alpha[t][:, None]
        b = beta[t]

        decayed = a * S                             # diag(a) S
        S = decayed - b * k @ (k.T @ decayed) + b * k @ v
        o = (S.T @ q).ravel()
        outs.append(o)

        if verbose:
            print(f"  token {t + 1}   k={K[t]}  v={V[t]}  beta={b:.2f}")
            show(("state S", S), f" -> o{t + 1} =", ("o", o), prec=2)
            print()

    return np.array(outs), S


def cost_per_token(d_k, d_v, s):
    """Per-token cost of one attention layer, KDA vs softmax, in FLOPs."""
    kda = 6 * d_k * d_v          # a handful of (d_k x d_v) ops, no s anywhere
    mla = 2 * s * (d_k + d_v)    # scores against s cached positions, then values
    return kda, mla


def short_conv(X, W, verbose=True, focus=0):
    """Depthwise causal 1D convolution along the SEQUENCE axis.

    X: (s, d)  -- s tokens, d channels (a projected q, k or v)
    W: (d, k)  -- one independent k-tap filter per channel

    Each channel is filtered separately: no mixing across channels, only across
    time. Causal, so position t sees only t-k+1 .. t (left-padded with zeros).
    """
    s, d = X.shape
    k = W.shape[1]
    Xpad = np.vstack([np.zeros((k - 1, d)), X])

    Y = np.zeros((s, d))
    for t in range(s):
        Y[t] = (Xpad[t : t + k] * W.T).sum(axis=0)

    if verbose:
        print(f"channel {focus} has its own {k} taps: {W[focus]}")
        print(f"it slides down the token axis and never sees another channel.\n")
        for t in range(s):
            win = Xpad[t : t + k, focus]
            terms = " + ".join(
                f"{w:.1f}*{('x' + str(t + 1 + i - (k - 1))) if t + i - (k - 1) >= 0 else '0'}"
                for i, w in enumerate(W[focus])
            )
            print(
                f"  t={t + 1}  window {np.array2string(win, precision=1, floatmode='fixed')}"
                f"   {terms}  =  {Y[t, focus]:6.2f}"
            )
    return Y


def params_by_hand(c):
    """Parameter count per component, derived from config numbers only.

    No torch, no module tree -- just the shapes written out. Cross-checks
    models.py: if the two disagree, one of them is wrong.
    """
    d, inner = c.d_model, c.kda_heads * c.kda_head_dim
    n_mla = sum(1 for x in _plan(c) if x == "mla")
    n_kda = c.n_layers - n_mla
    n_moe = c.n_moe_layers

    kda = (
        3 * d * inner                                   # W_q, W_k, W_v
        + d * inner                                     # W_g, full-rank gate
        + inner * d                                     # W_o
        + d * c.kda_alpha_rank + c.kda_alpha_rank * inner   # alpha low-rank
        + inner                                         # alpha bias
        + c.kda_heads                                   # per-head log scale
        + d * c.kda_heads                               # W_beta
        + 3 * inner * c.shortconv_kernel                # depthwise convs
        + c.kda_head_dim                                # head-wise RMSNorm
    )
    mla = (
        d * c.q_lora_rank
        + c.q_lora_rank * c.n_heads * c.qk_head_dim
        + d * (c.kv_lora_rank + c.qk_rope_head_dim)
        + c.kv_lora_rank * c.n_heads * (c.qk_nope_head_dim + c.v_head_dim)
        + c.q_lora_rank + c.kv_lora_rank                # the two RMSNorms
        + d * c.n_heads * c.v_head_dim                  # W_g
        + c.n_heads * c.v_head_dim * d                  # W_o
    )
    routed = c.n_routed * 3 * c.d_latent_moe * c.d_ff_expert
    shared = c.n_shared * 3 * d * c.d_ff_shared
    router = d * c.n_routed + c.n_routed + c.d_latent_moe   # router, its bias, the RMSNorm
    down_up = 2 * d * c.d_latent_moe

    return {
        "KDA": n_kda * kda,
        "Gated MLA": n_mla * mla,
        "MoE routed": n_moe * routed,
        "MoE shared": n_moe * shared,
        "MoE down/up": n_moe * down_up,
        "MoE router": n_moe * router,
        "Dense FFN": c.n_dense * 3 * d * c.d_ff_dense,
        "Embedding": c.d_vocab * d,
        "Output head": 0 if c.tie_embeddings else d * c.d_vocab,
        "Other": c.n_layers * 3 * d + d,                # 2 block norms + AttnRes query, + final norm
    }


def _plan(c):
    return [
        "mla" if (i % 4 == 0 or i == c.n_layers) else "kda"
        for i in range(1, c.n_layers + 1)
    ]
