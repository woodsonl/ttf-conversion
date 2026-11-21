"""ttfconv.bailing.convert end to end on a tiny random `bailing_hybrid` (CPU): the layer-major driver reads the
safetensors, folds, converts every layer and resumes. The float stream it carries through the folded layers must give
the model's own loss on the held-out windows, which checks the reader, the fold on a state dict, the one-layer shell
and the sorted expert forward against transformers' forward in one number."""
import json
import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("ttf.q38.moe")
from transformers import CONFIG_MAPPING

from ttfconv.arch import hf_class
from ttfconv.bailing import convert as C

SEQ, EVAL = 32, 2


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    if "bailing_hybrid" not in CONFIG_MAPPING:
        pytest.skip("transformers has no bailing_hybrid")
    from safetensors.torch import save_file
    d = tmp_path_factory.mktemp("bailing")
    # the Hub's field meanings: head_dim is the KDA head width (transformers' config moves it to linear_head_dim)
    json.dump(dict(hidden_size=128, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
                   num_key_value_heads=4, head_dim=32, qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
                   kv_lora_rank=32, layer_group_size=3, first_k_dense_replace=1, moe_intermediate_size=64,
                   num_local_experts=4, num_experts_per_tok=2, n_group=1, topk_group=1, vocab_size=512,
                   max_position_embeddings=512, pad_token_id=0, eos_token_id=1, kda_lower_bound=-5),
              open(d / "config.json", "w"))
    cfg = C.load_config(str(d))
    torch.manual_seed(0)
    m = hf_class("bailing_hybrid")(cfg).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("layernorm.weight") or n == "model.norm.weight":
                p.normal_(1.0, 0.2)                      # gains away from 1, so a fold that misses one shows
    save_file(m.state_dict(), str(d / "model.safetensors"))
    data = np.random.default_rng(0).integers(2, 512, 4096, dtype=np.uint32)
    data.tofile(d / "data.bin")
    ids = torch.from_numpy(data[-EVAL * SEQ:].astype(np.int64)).view(EVAL, SEQ)
    with torch.no_grad():
        ref = float(m(input_ids=ids, labels=ids).loss)
    return d, ref


def run(d, out, *extra):
    C.main([str(d), str(d / "data.bin"), str(out), "--windows", "4", "--seq", str(SEQ), "--eval-windows", str(EVAL),
            "--mb", "2", "--rank", "4", "--chunk", "2", "--experts-pass", "2", "--device", "cpu", *extra])


@pytest.mark.parametrize("extra", [(), ("--no-fold",), ("--pool", "1", "--frank", "4")])
def test_converts_and_resumes(model_dir, tmp_path, extra):
    d, ref = model_dir
    out = tmp_path / "out"
    run(d, out, "--layers", "3", *extra)
    assert not (out / "done").exists() and (out / "L2.pt").exists() and not (out / "L3.pt").exists()
    run(d, out, *extra)                                   # resumes at layer 3 from h.pt
    assert (out / "done").exists()
    ev = json.load(open(out / "eval.json"))
    assert abs(ev["float"] - ref) < 1e-3 * ref, (ev, ref)
    assert ev["ternary"] == ev["ternary"] and ev["ternary"] < 3 * ref
    L = torch.load(out / "L5.pt", weights_only=True)
    assert {"wpacked", "row_scale", "col_scale", "ta_packed"} <= set(L["mlp.experts.gate_up_proj"])
    assert "input_layernorm.weight" in L["_float"] and "mlp.gate.weight" in L["_float"]
    er = json.load(open(out / "L5.json"))
    assert 0.0 <= er["_router_agreement"] <= 1.0
    if "--frank" in extra:      # the float adapter is stored, and it lowers the error the ternary one left
        gu = er["mlp.experts.gate_up_proj"]
        assert L["mlp.experts.gate_up_proj"]["fa"].shape[-1] == 4 and gu["float"] <= gu["adapter"]
