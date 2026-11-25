"""Layer-streamed scores of the float BF16 model on token rows from exl3run/ids.py, one decoder layer on the GPU at a time.

Rows with "answer" (MMLU 5-shot, EXL3 eval/mmlu.py prompts): the pick is the argmax of the four choice tokens' logits at
the last prompt position. Other rows (wiki2, held-out windows): mean next-token loss and perplexity over every position,
the softmax over the first --vocab logits (EXL3 eval/ppl.py: the tokenizer's actual vocabulary).
The routed and shared experts' SwiGLU is clamped as the reference does (vLLM / SGLang, ExLlamaV3 act_limit):
min(silu(g), L) * clamp(u, -L, L) with L from expert_swiglu_limit_list / share_expert_swiglu_limit_list; transformers'
bailing_hybrid ignores the lists, and --no-limit keeps its unclamped form.
The residual stream of every row stays in host memory and is saved every --save layers to OUT.stream (resume), which is
deleted when OUT is written.
  python -m ttfconv.bailing.score HF_DIR IDS.safetensors OUT.json [--vocab 157153] [--tokens 16384] [--no-limit]
"""
import argparse, gc, json, math, os, time
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .convert import Reader, load_config, shell


def act(g, u, lim):
    s = F.silu(g)
    return s.clamp(max=lim) * u.clamp(-lim, lim) if lim else s * u


def experts_fwd(ex, lim):
    """BailingHybridExperts.forward with the tokens sorted by expert, and the clamp"""
    def fwd(x, idx, w):
        T, k = idx.shape
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        tok = torch.arange(T, device=x.device).repeat_interleave(k)[order]
        ws = w.reshape(-1)[order].to(x.dtype)
        offs = [0] + torch.bincount(flat, minlength=ex.num_experts).cumsum(0).tolist()
        xs, out = x[tok], torch.zeros_like(x)
        for e in range(ex.num_experts):
            a, b = offs[e], offs[e + 1]
            if b > a:
                g, u = F.linear(xs[a:b], ex.gate_up_proj[e]).chunk(2, -1)
                out.index_add_(0, tok[a:b], F.linear(act(g, u, lim), ex.down_proj[e]) * ws[a:b, None])
        return out
    return fwd


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("hf"); ap.add_argument("ids"); ap.add_argument("out")
    ap.add_argument("--vocab", type=int, default=0, help="softmax over the first VOCAB logits (0: all)")
    ap.add_argument("--tokens", type=int, default=16384, help="padded tokens per GPU batch")
    ap.add_argument("--save", type=int, default=6, help="save the stream every SAVE layers")
    ap.add_argument("--no-limit", action="store_true")
    a = ap.parse_args(argv)
    dev, dt = "cuda", torch.bfloat16
    log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)
    cfg, rd = load_config(a.hf), Reader(a.hf)
    nl = cfg.num_hidden_layers
    raw = json.load(open(os.path.join(a.hf, "config.json")))
    lim = [0] * nl if a.no_limit else raw.get("expert_swiglu_limit_list", [0] * nl)
    slim = [0] * nl if a.no_limit else raw.get("share_expert_swiglu_limit_list", [0] * nl)
    t = load_file(a.ids)
    ids = t["ids"]
    lens = t["len"] if "len" in t else torch.full((ids.shape[0],), ids.shape[1])
    batches, cur = [], []      # rows by length, as many per batch as fit --tokens padded
    for r in torch.argsort(lens).tolist():
        if cur and (len(cur) + 1) * int(lens[r]) > a.tokens:
            batches.append(cur); cur = []
        cur.append(r)
    batches.append(cur)
    batches = [(torch.tensor(b), int(lens[b].max())) for b in batches]
    sp = a.out + ".stream"
    if os.path.exists(sp):
        st = torch.load(sp, weights_only=True); first, hs = st["layer"], st["h"]
    else:
        emb = rd.get("model.word_embeddings.weight", "model.embed_tokens.weight")
        first, hs = 0, [emb[ids[b, :n]].to(dt) for b, n in batches]
        del emb
    log(f"{ids.shape[0]} rows, {int(lens.sum())} tokens, {len(batches)} batches; start at layer {first}; "
        f"limits {'off' if a.no_limit else 'on'}")
    for li in range(first, nl):
        t0 = time.time()
        m = shell(cfg, li)
        m.layers[0].load_state_dict(rd.layer(li), strict=True, assign=True)
        mlp = m.layers[0].mlp
        moe = hasattr(mlp, "experts")
        if moe:     # attention first with the routed experts in host memory: the reference linear attention needs the room
            ex = mlp.experts
            gu, dn = ex.gate_up_proj.data, ex.down_proj.data
            ex.gate_up_proj.data = ex.down_proj.data = torch.empty(0)
            mlp.forward = torch.zeros_like
        m.to(dev, dt)
        tl = time.time() - t0
        for h in hs:
            h.copy_(m(inputs_embeds=h.to(dev), use_cache=False).last_hidden_state.to("cpu"))
        if moe:     # then the MoE block on the attention output: h + mlp(post_attention_layernorm(h))
            del mlp.forward
            ex.gate_up_proj.data, ex.down_proj.data = gu.to(dev, dt), dn.to(dev, dt)
            del gu, dn
            ex.forward = experts_fwd(ex, lim[li])
            if slim[li]:
                sh = mlp.shared_experts
                sh.forward = lambda x, sh=sh, l=slim[li]: sh.down_proj(act(sh.gate_proj(x), sh.up_proj(x), l))
            ln = m.layers[0].post_attention_layernorm
            for h in hs:
                x = h.to(dev)
                h.copy_((x + mlp(ln(x))).to("cpu"))
            del ex, ln
        del m, mlp
        gc.collect()        # the experts' forward closures hold their modules in cycles: free the layer before the next
        torch.cuda.empty_cache()
        if (li + 1) % a.save == 0 and li + 1 < nl:
            torch.save({"layer": li + 1, "h": hs}, sp + ".tmp"); os.replace(sp + ".tmp", sp)
        log(f"layer {li}: load {tl:.0f}s, run {time.time() - t0 - tl:.0f}s; GPU peak "
            f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        torch.cuda.reset_peak_memory_stats()
    g = rd.get("model.norm.weight").float().to(dev)
    H = rd.get("lm_head.weight").float().to(dev)
    head = lambda x: (lambda x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * g)(x.float().to(dev))
    res = {"model": a.hf, "ids": a.ids, "limits": not a.no_limit}
    if "answer" in t:
        C = H[t["choice_ids"].to(dev)]
        pick, conf = torch.empty(ids.shape[0], dtype=torch.long), torch.empty(ids.shape[0])
        for (b, _), h in zip(batches, hs):
            z = head(h[torch.arange(len(b)), lens[b] - 1]) @ C.T
            pick[b] = z.argmax(-1).cpu()
            conf[b] = z.softmax(-1).cpu()[torch.arange(len(b)), t["answer"][b]]
        ok = pick == t["answer"]
        res.update(correct=int(ok.sum()), total=len(ok), accuracy=float(ok.float().mean()),
                   mean_correct_prob=float(conf.mean()), picks=pick.tolist())
    else:
        V = a.vocab or H.shape[0]
        nll, cnt = 0.0, 0
        for (b, n), h in zip(batches, hs):
            for i, r in enumerate(b.tolist()):
                z = head(h[i, :int(lens[r]) - 1]) @ H[:V].T
                nll += float(F.cross_entropy(z, ids[r, 1:int(lens[r])].to(dev), reduction="sum"))
                cnt += int(lens[r]) - 1
        res.update(loss=nll / cnt, perplexity=math.exp(nll / cnt), tokens=cnt)
    json.dump(res, open(a.out, "w"))
    if os.path.exists(sp):
        os.remove(sp)
    log({k: v for k, v in res.items() if k != "picks"})


if __name__ == "__main__":
    main()
