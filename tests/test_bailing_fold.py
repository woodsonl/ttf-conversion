"""The fold on a tiny `bailing_hybrid` (Ling-3.0), the family the converter is being taught.

The question is whether the residual-stream rotation is exact for a model whose two branch types both gate their
output: KDA multiplies its branch output by a sigmoid of a projection of the layer input (through the gated rmsnorm),
and every `layer_group_size`-th layer's MLA does the same head-wise. The rotation is exact only if those gates sit
*inside* the branch, upstream of the branch's only projection back to the residual width (`o_proj`, `dense`); a gate
applied to the residual stream itself would not commute with Q.

transformers ships this family natively from 5.23 (`BailingHybridConfig` / `BailingHybridForCausalLM`), in torch with
no fla dependency, so it runs on the CPU like the other families in test_arch.

Measured here: rotation exact, fold_err 3.3e-06, gain form `w` (the norm is plain, not zero-centred), three
structural kinds — a dense-MLP layer, KDA + MoE, gated MLA + MoE — and the layer-kind difference is visible in what
each attention reads off the residual stream: KDA reads six tensors (q, k, v, f, g, b), MLA three (q, kv_a, gate).
"""
import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import CONFIG_MAPPING

from ttfconv.arch import discover, hf_class


def small(**over):
    """a copy of the family's config at a size discovery runs on a CPU in seconds: hidden size and the head relations
    stay explicit, everything that only sets the cost comes down (as in test_arch)"""
    if "bailing_hybrid" not in CONFIG_MAPPING:
        pytest.skip("transformers has no bailing_hybrid")
    cfg = hf_class("bailing_hybrid", "config")()
    t = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    for k, v in dict(hidden_size=128, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
                     num_key_value_heads=4, linear_head_dim=32, head_dim=16, qk_nope_head_dim=32,
                     qk_rope_head_dim=16, v_head_dim=32, kv_lora_rank=32, layer_group_size=3,
                     first_k_dense_replace=1, moe_intermediate_size=64, num_local_experts=4,
                     num_experts_per_tok=2, n_group=1, topk_group=1, vocab_size=512,
                     max_position_embeddings=512, num_mtp_layers=0, pad_token_id=0, eos_token_id=1,
                     **over).items():
        if hasattr(t, k):
            setattr(t, k, v)
    return cfg


@pytest.fixture(scope="module")
def rules():
    return discover(small(), log=lambda s: None)


def test_fold_is_exact(rules):
    assert rules["rotation"], rules["reason"]
    assert rules["fold_err"] < 1e-4


def test_norm_gain_is_plain(rules):
    """BailingMoeV3RMSNorm is w, not the zero-centred 1 + w of the Qwen families: folding the wrong form leaves the
    gain in place and the rotation silently wrong"""
    forms = {seg["form"] for rule in rules["kinds"].values() for seg in rule}
    assert forms == {"w"}, forms


def test_both_attention_kinds_are_discovered(rules):
    """a tiny model that came out all-KDA would pass the logit check while saying nothing about the MLA layers, so
    assert the two kinds are both there: they differ in how many tensors they read off input_layernorm"""
    reads = {len(seg["in"]) for rule in rules["kinds"].values() for seg in rule
             if seg["norm"].endswith("input_layernorm")}
    assert {3, 6} <= reads, reads
