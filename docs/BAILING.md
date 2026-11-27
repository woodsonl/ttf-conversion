# bailing_hybrid (Ling-3.0-flash)

Ling-3.0-flash / flash-Sante: 42 layers + one MTP layer, KDA on every layer except each `layer_group_size`-th (6), which
is gated MLA; 512 routed experts + 1 shared per MoE layer, sigmoid `noaux_tc` router in fp32, untied head, vocab
157,184. 127.5 B parameters, 5.1 B active per token.

## transformers ships this family natively

`BailingHybridConfig` / `BailingHybridForCausalLM` are in transformers from 5.23, in torch
(`BailingHybridDeltaAttention` implements the gated delta rule itself and calls `fla`'s `chunk_kda` when
`flash-linear-attention` is installed). Architecture discovery, the fold check and the tiny-model tests run on a CPU
through the torch reference paths (`arch.cpu_reference_kernels`); nothing needs the model's remote modeling file.

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

`pytest -q tests/test_bailing_fold.py` (CPU, ~40 s) is the regression test: it asserts exactness, the `w` gain form, and
that both attention kinds were discovered. The full-size check runs in float32 against a 1e-4 tolerance, near which the
attention branch sits on this model; `bailing.convert.fold_rules` tries up to three seeds, and each rule it returns has
passed the same exact check.

## The converter

`python -m ttfconv.bailing.convert HF DATA OUT` runs layer-major over the native layers: one layer read from the
safetensors, folded, its dense matrices and its routed experts (router-weighted input moments, `--experts-pass` experts
held at a time) rounded with `q38`'s `convert_group`, written to `OUT/L{i}.pt`, and its fake-quantised weights written
back so the next layer reads the converted stream. A stopped run resumes at the first missing `L{i}.pt`. The MTP layer
is dropped. Options used below: `--pool` (shrink each expert's input covariance toward the pooled MoE input), `--frank`
(closed-form float adapter of that rank beside every ternary matrix), `--router-fit` (closed-form router refit). The
model's SwiGLU clamp (`expert_swiglu_limit_list`, `share_expert_swiglu_limit_list`) is applied as the checkpoint
specifies it; a null list means no clamp. On CUDA the converter puts transformers' `fla` kernel wrappers back after the
fold rules are built: on one 2,048-token window the torch reference KDA allocates 6.25 GiB against `fla`'s 0.30 GiB.

## Ling-3.0-flash-Sante on one RTX 5080

One RTX 5080 (16 GB), 123 GB host memory. Calibration on `wiki.bin`, sequences of 2,048, `--mb 1 --experts-pass 64
--chunk 16`, rank 16, seed 777. Held-out loss in nats on the last 16,384 tokens of `wiki.bin` (8 windows of 2,048), the
float model through the same path at 1.406 to 1.408 in every run. "Float routing" feeds each converted MoE layer the
float model's routing decisions.

| run | settings | ternary | float routing | wall time |
|---|---|---|---|---|
| base | 64 windows | 2.236 | 2.151 | 59 min |
| pf64 | 64 windows, `--pool 1 --frank 16` | 2.172 | 2.094 | 62 min |
| pf64r | repeat of pf64, stopped after layer 21 and resumed | 2.155 | 2.077 | |
| rf64 | pf64 + `--router-fit` | 2.181 | 2.094 | 64 min |
| w512 | 512 windows | 2.102 | 2.035 | 104 min |
| w512pf | 512 windows, `--pool 1 --frank 16` | 2.032 | 1.983 | 107 min |

Peak GPU memory 12.6 GiB on the MoE layers. The router refit moves the router weights 19% to 42% (relative norm) and
keeps routing agreement at layers 32 to 41 between 0.70 and 0.74; its loss falls inside the pf64 / pf64r spread (0.017).
Float routing recovers 0.078 (pf64r) and 0.087 (rf64) of a 0.75 to 0.78 gap to float, so routing accounts for about a
tenth of the loss.

## What is still open

- **The export emitter.** `ttf.export` has no emitter or check for this architecture.
- **Recovery at full size.** Only Ling-3.0-tiny has recovery runs (`ttf-ternary`, `docs/BAILING.md`).
- **Downstream evaluation** of the converted model beyond held-out loss.
