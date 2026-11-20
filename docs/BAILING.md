# bailing_hybrid (Ling-3.0-flash)

Ling-3.0-flash / flash-Sante: 42 layers + one MTP layer, KDA on every layer except each `layer_group_size`-th (6), which
is gated MLA; 512 routed experts + 1 shared per MoE layer, sigmoid `noaux_tc` router in fp32, untied head, vocab
157,184. 127.5 B parameters, 5.1 B active per token.

## transformers ships this family natively

`BailingHybridConfig` / `BailingHybridForCausalLM` are in transformers from 5.23, in torch, with no `fla` dependency
(`BailingHybridDeltaAttention` implements the gated delta rule itself). Architecture discovery, the fold check and the
tiny-model tests therefore run on a CPU; nothing needs the model's remote modeling file, and nothing needs Triton.

Two config details that cost an hour each if missed:

- `head_dim` is **not** the KDA head width. `__init__` copies it to `linear_head_dim` and then overwrites
  `head_dim = qk_rope_head_dim`. Set both when building a small instance; setting `head_dim` alone leaves RoPE with the
  wrong frequency count and the attention fails on a shape mismatch.
- `layer_types` and `mlp_layer_types` are derived at construction from `layer_group_size` and `first_k_dense_replace`,
  so shortening `num_hidden_layers` afterwards means truncating both lists (`arch.tiny_config` does).

## The residual rotation is exact

`ttfconv.arch.discover` on a tiny instance: `rotation=True`, `fold_err = 3.3e-06`, and the norm gain form is `w` (the
norm is plain, not the zero-centred `1 + w` of the Qwen families).

The gates are what made this doubtful, and they turn out not to obstruct anything. KDA multiplies its branch output by
`sigmoid(g_proj(x))` through the gated rmsnorm, and gated MLA multiplies its attention output by
`sigmoid(g_proj(x))` head-wise, but both sit **inside** the branch and upstream of the branch's only projection back to
the residual width (`o_proj`, `o_proj`, `dense`). The rotation acts on that projection's output, so no gate ever sees
the rotated stream.

The discovered rules split into three structural kinds, and the attention kinds are distinguishable by how much they
read off the residual stream:

| Layer kind | reads `input_layernorm` | writes | reads `post_attention_layernorm` |
|---|---|---|---|
| dense MLP (layer 0) | 6 — q, k, v, f, g, b (KDA) | o_proj | 2 — gate, up |
| KDA + MoE | 6 — q, k, v, f, g, b | o_proj | 4 — router, gate, up, experts |
| gated MLA + MoE | 3 — q, kv_a, g | dense | 4 |

So no fold-free fallback is needed for this family: the conversion can take `--ext ttfconv.fold` like the 27B did, and
the float adapters (`--ext ttfconv.adapters`) are an addition on top of a folded model rather than a substitute for one.

`pytest -q tests/test_bailing_fold.py` (CPU, ~40 s) is the regression test: it asserts exactness, the `w` gain form, and
that both attention kinds were discovered.

## What the conversion still needs

Anchors are in place (`ttfconv/arch.py` discovery, `ttfconv/gptq.py` rounding, `ttfconv/adapters.py`), but no converter
exists for this family yet:

- **The model builder.** ttfconv builds models through ttf-core's Qwen wrapper. The native `BailingHybrid` stores its
  experts fused (`mlp.experts.gate_up_proj` of shape `[num_experts, out, in]`), where the Ling checkpoint on the Hub
  keeps `experts.N.{gate,up,down}_proj` per expert. Pick one layout and make the reader, the per-layer stream and the
  trainer agree on it; the converter's own `[1536, 2560]` / `[2560, 768]` shapes are what `ttf.q38.moe` already
  handles.
- **The stream driver.** `ttfconv/stream.py` is not family-specific in its structure (layer inputs stashed, one layer
  resident, Hessians per layer), but it assumes `hf.model.layers` and a swappable module path.
- **The MTP layer** (index 42, 3.07 B parameters, 2.4% of the model). Drop it; converting a speculative head buys
  nothing and the trainer will not load it.
- **The export emitter.** `ttf.export` has no emitter or check for this architecture.

Per-layer cost is friendlier than the 27B in one respect: an expert's K is 2560 against the 27B's 5120, so per-parameter
GPTQ work halves. What goes up is matrix count (42,359 of them, most of them `[768, 2560]`), which makes small-matrix
latency rather than arithmetic the risk.
