Kimi K3 Exercise
================

Working through the Kimi K3 technical report, focused on inference FLOPs.

Getting Started
---------------

Install dependencies

```bash
uv sync
```

Then either open up the `kimik3.ipynb` file in your IDE or use

```bash
uv run --with jupyter jupyter lab
```

Contents
--------

| file | what it is |
|---|---|
| `kimik3.ipynb` | the notebook — Table 1, param validation, FLOPs from first principles |
| `models.py` | K3 skeleton built on `torch.device("meta")`; shapes only, no storage |
| `flops.py` | matmul/conv/KDA display helpers and a hand-derived parameter count |
| `k3-layers.html` | schematics of every matmul in a KDA / Gated MLA / LatentMoE layer |
| `k3_config.json` | config pulled from `moonshotai/Kimi-K3` |

What's checked
--------------

- 2.78T total and 104.2B active params reproduced to within 0.02% by building the
  model on the meta device and counting.
- `layer_plan()` reproduces the config's `full_attn_layers` and `kda_layers`
  index-for-index (69 KDA, 24 Gated MLA).
- Hand-derived parameter counts (`flops.params_by_hand`) agree with the module
  tree component-by-component, to the parameter.
- The numpy ShortConv matches `torch.nn.Conv1d` with `groups=channels`.

References
----------
Kimi K3 — `2607.24653v2.pdf`

Kimi Linear (KDA) — `2510.26692.pdf` — https://arxiv.org/abs/2510.26692

Muon tutorial — ../muon-tutorial
