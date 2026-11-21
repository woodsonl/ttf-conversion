"""Ling-3.0 (HF model_type bailing_hybrid) -> ternary, layer by layer.

The model is transformers' own BailingHybrid (5.23+), one decoder layer resident at a time. Per layer:
  1. the layer's weights are read from the safetensors (the Hub's names and per-expert tensors, or transformers' own);
  2. the fold (ttfconv.arch's discovered rule: each residual norm's gain into its readers, then the shared rotation Q
     on every residual axis), unless --no-fold;
  3. the calibration windows go through the float layer and every ternary matrix collects its input moments (the routed
     experts' weighted by their router weight), --experts-pass experts at a time so the [E, K, K] moments fit the device;
  4. each matrix is rounded by ttfconv.q38.convert's convert_group (centred GPTQ, row / column scales, a rank --rank
     ternary adapter) into the TernaryExperts form (ttf.q38.moe);
  5. the windows go through the converted (fake-quantised) layer to become the next layer's inputs.
Ternary: the routed experts' gate+up and down, and every linear wider than 64 outputs. Float: the norms, the depthwise
convolution, the forget gate's A_log / dt_bias, the router, and the head-count-wide gates (b_proj, the MLA g_proj).
The MTP layer is not part of transformers' model and is not read.

Output: OUT/L{i}.pt = {name: TernaryExperts state_dict, "_float": the layer's remaining tensors, folded}, OUT/L{i}.json
(relative output errors), OUT/h.pt (the next layer's inputs; resume restarts at the first missing layer), OUT/meta.json
and, when all layers are done, OUT/eval.json: the loss on held-out windows of the float model and of the converted one.
  python -m ttfconv.bailing.convert HF_DIR DATA.bin OUT [--windows 64] [--rank 16] [--no-fold] [--device cpu]
"""
import argparse, copy, json, os, re, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..arch import discover, tiny_model, decoder_layers, signature, hf_class, _scale, _turn
from ..fold import residual_q

FLOAT_OUT = 64      # a linear with at most this many outputs stays float (b_proj, the MLA head gate)
RENAME = ((r"^attention\.", "self_attn."),
          (r"^self_attn\.(f_proj\.|dt_bias$|A_log$)", r"self_attn.forget_gate.\1"),
          (r"^self_attn\.dense\.", "self_attn.o_proj."),
          (r"^mlp\.gate\.expert_bias$", "mlp.gate.e_score_correction_bias"),
          # q_lora_rank (Ling-3.0-tiny): the MLA query is q_b(norm(q_a(x))), held here as one q_proj (see shell)
          (r"^self_attn\.q_a_proj\.", "self_attn.q_proj.a."), (r"^self_attn\.q_a_layernorm\.", "self_attn.q_proj.norm."),
          (r"^self_attn\.q_b_proj\.", "self_attn.q_proj.b."))


def load_config(hf):
    raw = json.load(open(os.path.join(hf, "config.json")))
    for k in ("auto_map", "architectures", "model_type", "transformers_version", "torch_dtype"):
        raw.pop(k, None)
    if "kda_lower_bound" in raw:        # the Hub's config stores an int, transformers' config validates a float
        raw["kda_lower_bound"] = float(raw["kda_lower_bound"])
    return hf_class("bailing_hybrid", "config")(**raw)


class Reader:
    """tensors of an HF directory by name, and one decoder layer's state dict in transformers' names"""

    def __init__(self, hf):
        from safetensors import safe_open
        self.hf, self.open, self.files = hf, safe_open, {}
        idx = os.path.join(hf, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.map = json.load(open(idx))["weight_map"]
        else:
            self.map = {k: "model.safetensors" for k in self._file("model.safetensors").keys()}

    def _file(self, f):
        if f not in self.files:
            self.files[f] = self.open(os.path.join(self.hf, f), "pt")
        return self.files[f]

    def get(self, *names):
        k = next(n for n in names if n in self.map)
        return self._file(self.map[k]).get_tensor(k)

    def layer(self, li):
        pre = f"model.layers.{li}."
        sd = {}
        for k in self.map:
            if k.startswith(pre):
                n = k[len(pre):]
                for a, b in RENAME:
                    n = re.sub(a, b, n)
                sd[n] = self.get(k)
        ex = sorted({int(m.group(1)) for m in (re.match(r"mlp\.experts\.(\d+)\.", n) for n in sd) if m})
        if ex:                           # the Hub keeps one module per expert; transformers keeps them fused
            pop = lambda leaf: torch.stack([sd.pop(f"mlp.experts.{e}.{leaf}.weight") for e in ex])
            sd["mlp.experts.gate_up_proj"] = torch.cat([pop("gate_proj"), pop("up_proj")], 1)
            sd["mlp.experts.down_proj"] = pop("down_proj")
        if "self_attn.q_conv1d.weight" in sd:
            sd["self_attn.conv1d.weight"] = torch.cat([sd.pop(f"self_attn.{c}_conv1d.weight") for c in "qkv"], 0)
        return sd


def shell(cfg, li):
    """transformers' model holding only decoder layer li (its masks, position embeddings and layer call as the library
    does them), with no embedding to speak of and no final norm: inputs_embeds in, the layer's output out"""
    from transformers.models.bailing_hybrid.modeling_bailing_hybrid import BailingHybridModel
    c = copy.deepcopy(cfg)
    c.layer_types, c.mlp_layer_types = [cfg.layer_types[li]], [cfg.mlp_layer_types[li]]
    c.num_hidden_layers, c.vocab_size, c.pad_token_id = 1, 8, 0
    m = BailingHybridModel(c)       # ponytail: random init of a layer that is overwritten next; skip it if it shows
    r, att = getattr(cfg, "q_lora_rank", None), m.layers[0].self_attn
    if r and cfg.layer_types[li] == "full_attention":   # transformers' attention has the full-rank query only
        from collections import OrderedDict
        from transformers.models.bailing_hybrid.modeling_bailing_hybrid import BailingHybridRMSNorm
        att.q_proj = nn.Sequential(OrderedDict(a=nn.Linear(cfg.hidden_size, r, bias=False), norm=BailingHybridRMSNorm(r),
                                               b=nn.Linear(r, att.q_proj.out_features, bias=False)))
    m.norm = nn.Identity()
    return m.eval()


def fold_rules(cfg):
    """{(layer type, mlp type): the layer's fold rule}. ttfconv.arch keys its rules by the parameter shapes of its
    tiny model, which the full-size layers do not share; the layer kind is what carries over."""
    r = discover(cfg, log=lambda s: None)
    assert r["rotation"], r["reason"]
    tm = tiny_model(cfg)
    tc = tm.config
    kinds = {(tc.layer_types[i], tc.mlp_layer_types[i]): r["kinds"][signature(L)]
             for i, L in enumerate(decoder_layers(tm)[1])}
    if getattr(cfg, "q_lora_rank", None):               # the residual reader of a low-rank query is its first factor
        for (lt, _), rule in kinds.items():
            for seg in rule:
                for e in seg["in"]:
                    if lt == "full_attention" and e[0] == "self_attn.q_proj.weight":
                        e[0] = "self_attn.q_proj.a.weight"
    return kinds


@torch.no_grad()
def fold_state(sd, rule, Q, dev, chunk=32):
    """ttfconv.arch.fold_layer on a state dict, the 3-D expert tensors `chunk` experts at a time on dev"""
    Q = Q.to(dev)
    for seg in rule:
        nk = seg["norm"] + ".weight"
        g = sd[nk].float().to(dev)
        todo = {}
        for pn, ax in seg["in"]:
            todo.setdefault(pn, []).append(("in", ax))
        for pn, ax in seg["out"]:
            todo.setdefault(pn, []).append(("out", ax))
        for pn, ops in todo.items():
            p = sd[pn]
            step = chunk if p.dim() == 3 else p.shape[0]     # slices along the expert axis keep the rule's axes
            for e0 in range(0, p.shape[0], step):
                t = p[e0:e0 + step].float().to(dev)
                for kind, ax in ops:
                    if kind == "in":
                        t = _scale(t, ax, g)
                    t = _turn(t, ax, Q)
                p[e0:e0 + step] = t.to(p.dtype).cpu()
        sd[nk] = torch.ones_like(sd[nk])


class RangeStats:
    """ttfconv.q38.convert.Stats over experts [e0, e1) only: the [E, K, K] second moments of all experts at once do not
    fit a 16 GB device at K = 2560"""

    def __init__(self, E, K, dev, chunk, e0, e1):
        from ..q38.convert import Stats
        assert e0 % chunk == 0
        s = self.st = Stats.__new__(Stats)
        s.E, s.K, s.dev, s.chunk = E, K, dev, chunk
        s.H = [torch.zeros(min(chunk, E - a), K, K, device=dev) if e0 <= a < e1 else None for a in range(0, E, chunk)]
        s.s = torch.zeros(E, K, device=dev); s.n = torch.zeros(E, device=dev)
        s.cnt = torch.zeros(E, dtype=torch.long, device=dev)
        self.r = (e0, e1)

    def add(self, x, offs, w=None):
        e0, e1 = self.r
        self.st.add(x, [offs[e0]] * (e0 + 1) + offs[e0 + 1:e1 + 1] + [offs[e1]] * (self.st.E - e1), w)


def experts_forward(ex, x, idx, w, cap=None, bias=None):
    """BailingHybridExperts.forward with the tokens sorted by expert: cap(xs, mid, offs, ws) sees every expert's
    inputs, its down projection's inputs and the router weights; bias = (gate+up [E, 2I], down [E, d]) or None"""
    T, k = idx.shape
    E = ex.num_experts
    flat = idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    tok = torch.arange(T, device=x.device).repeat_interleave(k)[order]
    ws = w.reshape(-1)[order].to(x.dtype)
    offs = [0] + torch.bincount(flat, minlength=E).cumsum(0).tolist()
    xs = x[tok]
    mid = torch.empty(xs.shape[0], ex.intermediate_dim, dtype=x.dtype, device=x.device)
    out = torch.zeros_like(x)
    for e in range(E):
        a, b = offs[e], offs[e + 1]
        if b > a:
            gate, up = F.linear(xs[a:b], ex.gate_up_proj[e], None if bias is None else bias[0][e]).chunk(2, -1)
            mid[a:b] = ex.act_fn(gate) * up
            o = F.linear(mid[a:b], ex.down_proj[e], None if bias is None else bias[1][e])
            out.index_add_(0, tok[a:b], o * ws[a:b, None])
    if cap is not None:
        cap(xs, mid, offs, ws)
    return out


@torch.no_grad()
def convert_layer(m, a, h, hq, hf, dev, dt):
    """m: shell with the (folded) float layer loaded. Returns (state dict, errors). In place: h (the calibration
    windows, which alone feed the statistics) and hq (held-out windows) advance through the converted layer, hf (the
    held-out windows' float stream) through the float one."""
    from ..q38.convert import Stats, convert_group, to_state, _cat_state, _err_summary
    from ttf.q38.moe import pack3
    L = m.layers[0]
    run = lambda x: m(inputs_embeds=x.to(dev, dt), use_cache=False).last_hidden_state

    def advance(s):
        for b0 in range(0, s.shape[0], a.mb):
            s[b0:b0 + a.mb] = run(s[b0:b0 + a.mb]).to("cpu", s.dtype)
    dense = [(n, mod) for n, mod in L.named_modules() if isinstance(mod, nn.Linear) and mod.out_features > FLOAT_OUT]
    S = {n: Stats(1, lin.in_features, dev) for n, lin in dense}
    moe = hasattr(L.mlp, "experts")
    sd, errs = {}, {}
    if moe:
        ex = L.mlp.experts
        E, I, d = ex.num_experts, ex.intermediate_dim, ex.hidden_dim
        step = max(a.chunk, (min(a.experts_pass, E) // a.chunk) * a.chunk)
        Sdn = Stats(E, I, dev)
        What_gu = torch.empty(E, 2 * I, d, dtype=dt); bias_gu = torch.empty(E, 2 * I, dtype=dt)
        parts, eg, cnt = [], {"main": [], "adapter": []}, 0
    for p0 in (range(0, E, step) if moe else [0]):
        first = p0 == 0
        hooks = []
        if first:
            hooks = [lin.register_forward_pre_hook(
                (lambda nm: lambda mod, inp: S[nm].add(inp[0].reshape(-1, inp[0].shape[-1]),
                                                       [0, inp[0].numel() // inp[0].shape[-1]]))(n)) for n, lin in dense]
        if moe:
            Sgu = RangeStats(E, d, dev, a.chunk, p0, min(p0 + step, E))

            def cap(xs, mid, offs, ws, Sgu=Sgu, first=first):
                Sgu.add(xs, offs, ws)
                if first:
                    Sdn.add(mid, offs, ws)
            ex.forward = lambda x, idx, w, cap=cap: experts_forward(ex, x, idx, w, cap)
        for b0 in range(0, h.shape[0], a.mb):
            run(h[b0:b0 + a.mb])
        for hk in hooks:
            hk.remove()
        if moe:
            ex.forward = lambda x, idx, w: experts_forward(ex, x, idx, w)
        if first:
            advance(hf)
        if not moe:
            break
        st = Sgu.st
        for e0 in range(p0, min(p0 + step, E), a.chunk):
            e1 = min(e0 + a.chunk, E)
            W = ex.gate_up_proj[e0:e1].float()
            Hc, mu, _ = st.centred(e0, e1)
            st.H[e0 // a.chunk] = None
            o, Wh, er = convert_group(W, Hc, mu, a.rank, 0.05 if int(st.cnt[e0:e1].min()) < 2 * d else 0.01)
            What_gu[e0:e1] = Wh.to("cpu", dt); bias_gu[e0:e1] = torch.einsum("enk,ek->en", W - Wh, mu).to("cpu", dt)
            parts.append({k: v.cpu() for k, v in o.items()})
            eg["main"].append(er["main"].cpu()); eg["adapter"].append(er["adapter"].cpu())
            del W, Hc, Wh, o
        cnt = cnt + st.cnt.cpu()
        del Sgu, st
    if moe:
        sd["mlp.experts.gate_up_proj"] = _cat_state(parts, pack3)
        errs["mlp.experts.gate_up_proj"] = _err_summary(eg, cnt)
        W = ex.down_proj.float()
        Hc, mu, _ = Sdn.centred(0, E)
        o, Wh, er = convert_group(W, Hc, mu, a.rank, 0.01)
        sd["mlp.experts.down_proj"] = to_state(o, pack3)
        errs["mlp.experts.down_proj"] = _err_summary({"main": [er["main"]], "adapter": [er["adapter"]]}, Sdn.cnt)
        bias = (bias_gu.to(dev), torch.einsum("enk,ek->en", W - Wh, mu).to(dt))
        ex.down_proj.data = Wh.to(dt)
        ex.gate_up_proj.data = What_gu.to(dev)
        ex.forward = lambda x, idx, w: experts_forward(ex, x, idx, w, None, bias)
        del W, Hc, Wh, Sdn, What_gu
    for n, lin in dense:
        W = lin.weight.float()[None]
        Hc, mu, _ = S[n].centred(0, 1)
        o, Wh, er = convert_group(W, Hc, mu, a.rank, 0.01)
        sd[n] = to_state(o, pack3)
        errs[n] = {"main": float(er["main"][0]), "adapter": float(er["adapter"][0])}
        lin.weight.data = Wh[0].to(dt)
        lin.bias = nn.Parameter(torch.einsum("enk,ek->en", W - Wh, mu)[0].to(dt), requires_grad=False)
        del S[n]
    advance(h); advance(hq)
    return sd, errs


@torch.no_grad()
def eval_loss(rd, cfg, hs, ids, Q, dev, mb):
    """mean next-token loss of residual streams hs [n, seq, d] (in the rotated basis when Q is given)"""
    g = rd.get("model.norm.weight").float().to(dev)
    H = rd.get("lm_head.weight").float().to(dev)
    tot = 0.0
    for b0 in range(0, hs.shape[0], mb):
        x = hs[b0:b0 + mb].float().to(dev)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
        if Q is not None:
            x = x @ Q.to(dev)
        logits = (x * g) @ H.T
        y = ids[b0:b0 + mb].to(dev)
        tot += float(F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), y[:, 1:].reshape(-1),
                                     reduction="sum"))
    return tot / (hs.shape[0] * (hs.shape[1] - 1))


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("hf"); ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--windows", type=int, default=64); ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--eval-windows", type=int, default=8, help="held-out windows, from the end of the data")
    ap.add_argument("--mb", type=int, default=4); ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--layers", type=int, default=0); ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--chunk", type=int, default=32, help="experts per GPTQ batch")
    ap.add_argument("--experts-pass", type=int, default=128, help="experts whose input moments are held at once")
    ap.add_argument("--no-fold", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = a.device
    dt = torch.bfloat16 if dev.startswith("cuda") else torch.float32
    os.makedirs(a.out, exist_ok=True)
    log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)
    cfg = load_config(a.hf)
    nl = a.layers or cfg.num_hidden_layers
    meta = {k: getattr(a, k) for k in ("windows", "seq", "eval_windows", "rank", "seed", "no_fold")}
    mp = os.path.join(a.out, "meta.json")
    if os.path.exists(mp):
        assert json.load(open(mp)) == meta, "OUT was started with other settings"
    json.dump(meta, open(mp, "w"))
    first = next((i for i in range(nl) if not os.path.exists(os.path.join(a.out, f"L{i}.pt"))), nl)
    if first == nl and (nl < cfg.num_hidden_layers or os.path.exists(os.path.join(a.out, "done"))):
        log("all layers converted"); return
    rd = Reader(a.hf)
    Q = rules = None
    if not a.no_fold:
        rules = fold_rules(cfg)
        Q = residual_q(cfg.hidden_size, torch.Generator().manual_seed(a.seed))
    data = np.memmap(a.data, dtype=np.uint32, mode="r")
    held = a.eval_windows * a.seq
    starts = np.random.default_rng(a.seed).integers(0, len(data) - held - a.seq - 1, a.windows)
    take = lambda ss: torch.from_numpy(np.stack([data[s:s + a.seq].astype(np.int64) for s in ss]))
    ids, ids_e = take(starts), take(range(len(data) - held, len(data), a.seq))
    hp = os.path.join(a.out, "h.pt")
    if first == 0:
        emb = rd.get("model.embed_tokens.weight", "model.word_embeddings.weight")
        look = lambda i: (emb[i].float() @ Q.T if Q is not None else emb[i].float()).to(dt)
        hs = {"h": look(ids), "hq": look(ids_e)}
        hs["hf"] = hs["hq"].clone()
        del emb
    else:
        hs = torch.load(hp, weights_only=True)
        assert int(open(os.path.join(a.out, "h.layer")).read()) == first
    log(f"start at layer {first} of {nl}: {a.windows} windows x {a.seq}, fold {not a.no_fold}, {dev}")
    for li in range(first, nl):
        t0 = time.time()
        sd = rd.layer(li)
        if rules is not None:
            fold_state(sd, rules[(cfg.layer_types[li], cfg.mlp_layer_types[li])], Q, dev, a.chunk)
        m = shell(cfg, li)
        m.layers[0].load_state_dict(sd, strict=True)
        m.to(dev, dt)
        del sd
        out, errs = convert_layer(m, a, hs["h"], hs["hq"], hs["hf"], dev, dt)
        out["_float"] = {k: v.cpu() for k, v in m.layers[0].state_dict().items()
                         if k not in out and k.rsplit(".", 1)[0] not in out}
        del m
        torch.save(hs, hp + ".tmp"); os.replace(hp + ".tmp", hp)
        open(os.path.join(a.out, "h.layer"), "w").write(str(li + 1))
        tmp = os.path.join(a.out, f"L{li}.pt.tmp")
        torch.save(out, tmp); os.replace(tmp, os.path.join(a.out, f"L{li}.pt"))
        json.dump(errs, open(os.path.join(a.out, f"L{li}.json"), "w"), indent=1)
        peak = f"; GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if dev.startswith("cuda") else ""
        log(f"layer {li} ({cfg.layer_types[li]}, {cfg.mlp_layer_types[li]}): {time.time() - t0:.0f}s; rel out err "
            + " ".join(f"{n.split('.')[-1]} {v['main']:.3f}->{v['adapter']:.3f}" for n, v in errs.items())
            + f"; h rms {float(hs['h'][:2].float().pow(2).mean().sqrt()):.3f}" + peak)
        if dev.startswith("cuda"):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    if nl == cfg.num_hidden_layers:
        ev = {"float": eval_loss(rd, cfg, hs["hf"], ids_e, Q, dev, a.mb),
              "ternary": eval_loss(rd, cfg, hs["hq"], ids_e, Q, dev, a.mb),
              "calibration_ternary": eval_loss(rd, cfg, hs["h"], ids, Q, dev, a.mb)}
        json.dump(ev, open(os.path.join(a.out, "eval.json"), "w"), indent=1)
        log("held-out loss: float %.4f, ternary %.4f" % (ev["float"], ev["ternary"]))
        open(os.path.join(a.out, "done"), "w").close()


if __name__ == "__main__":
    main()
