"""
Etude ResNet-18 (phase 2) qui SUIT le rapport de reference (chap. 5.2) :
  5.2.2  choix du critere d'elagage : L1 / L2 / activation moyenne / Taylor  (F1 vs sparsite, elague seul vs distille)
  5.2.3  la distillation transfere-t-elle de la connaissance ?  (KD  vs  entrainement CE simple, meme budget)
  5.2.4  role de la re-initialisation des poids avant la distillation  (Kaiming vs poids conserves)
  5.2.5  quantification INT8 statique seule (FP32 vs INT8)
  5.2.6  pipeline complet P-KD-Q : baseline -> elague (Taylor) -> distillation -> INT8   (4 etapes)
Datasets : coco7 (7 classes COCO, recadrages de boites, comme le rapport) et imagenette ; "synthetic" = test de fumee.

Ecarts DECLARES par rapport au rapport (a mettre dans le texte) :
  - elagage STRUCTURE par canaux avec graphe de dependances Torch-Pruning, classement global (scores normalises
    min-max par groupe apres ecretage au 99e percentile), au lieu de masques non structures ;
  - la "sparsite" est exprimee en REDUCTION DE PARAMETRES (recherche dichotomique du ratio de canaux) ;
  - images redimensionnees a --img-size (128 par defaut) et nombre d'epoques reduit (CPU i5, pas de GPU) ;
  - pas de teacher ViT-B/32 (trop lourd sur CPU) : le controle "KD vs CE" utilise le meme ResNet-18 comme teacher.

Commandes (une ligne chacune) :
  python run_resnet18_study.py prepare --dataset coco7 --coco-images D:\\coco\\val2017 --coco-ann D:\\coco\\annotations\\instances_val2017.json
  python run_resnet18_study.py prepare --dataset imagenette          (telecharge ~100 Mo)
  python run_resnet18_study.py all --dataset coco7                    (enchaine tout, reprend ou il s'est arrete)
  python run_resnet18_study.py report                                  (figures + tableaux .tex + section LaTeX)
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path("r18_report")
RES = OUT / "results"
FIG = OUT / "figures"
TAB = OUT / "tables"
COCO7 = ["airplane", "bus", "horse", "zebra", "banana", "motorcycle", "laptop"]
CRITERIA = ["l1", "l2", "activation", "taylor"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def jdump(name, obj):
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{name}.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")


def jload(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


# ============================================================================ donnees
def prepare(args):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    S = args.img_size
    if args.dataset == "synthetic":
        n = 600
        classes = [f"c{i}" for i in range(5)]
        y = rng.randint(0, 5, n)
        x = (rng.rand(n, S, S, 3) * 60).astype(np.int32)
        for i in range(n):
            x[i, :, :, y[i] % 3] += 120
            x[i, : S // 2, :, :] += (y[i] * 20) % 90
        x = np.clip(x, 0, 255).astype(np.uint8)
        cut = int(0.8 * n)
        np.savez_compressed(OUT / f"data_synthetic_{S}.npz", xtr=x[:cut], ytr=y[:cut], xva=x[cut:], yva=y[cut:], classes=classes)
        return
    from PIL import Image
    if args.dataset == "coco7":
        ann = json.loads(Path(args.coco_ann).read_text(encoding="utf-8"))
        cat = {c["id"]: c["name"] for c in ann["categories"] if c["name"] in COCO7}
        ids = {i["id"]: i["file_name"] for i in ann["images"]}
        items = {k: [] for k in COCO7}
        for a in ann["annotations"]:
            if a["category_id"] in cat and not a.get("iscrowd", 0) and min(a["bbox"][2:]) >= args.min_box:
                items[cat[a["category_id"]]].append((a["image_id"], a["bbox"]))
        xtr, ytr, xva, yva = [], [], [], []
        for ci, name in enumerate(COCO7):
            lst = items[name]
            rng.shuffle(lst)
            for img_id, (bx, by, bw, bh) in lst:
                split_val = (img_id % 5 == 0)                       # split 80/20 PAR IMAGE (comme le rapport)
                tgt_x, tgt_y, cap = (xva, yva, args.max_val) if split_val else (xtr, ytr, args.max_train)
                if sum(1 for t in tgt_y if t == ci) >= cap:
                    continue
                p = Path(args.coco_images) / ids[img_id]
                if not p.is_file():
                    continue
                im = Image.open(p).convert("RGB").crop((bx, by, bx + bw, by + bh)).resize((S, S), Image.BILINEAR)
                tgt_x.append(np.asarray(im))
                tgt_y.append(ci)
            log(f"  {name}: train={sum(1 for t in ytr if t == ci)} val={sum(1 for t in yva if t == ci)}")
        classes = COCO7
    elif args.dataset == "imagenette":
        import tarfile
        import urllib.request
        root = OUT / "imagenette2-160"
        if not root.is_dir():
            tgz = OUT / "imagenette2-160.tgz"
            log("telechargement imagenette2-160 ...")
            urllib.request.urlretrieve("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz", tgz)
            with tarfile.open(tgz) as t:
                t.extractall(OUT)
        classes = sorted(d.name for d in (root / "train").iterdir() if d.is_dir())
        xtr, ytr, xva, yva = [], [], [], []
        for split, X, Y, cap in (("train", xtr, ytr, args.max_train), ("val", xva, yva, args.max_val)):
            for ci, c in enumerate(classes):
                files = sorted((root / split / c).glob("*.JPEG"))
                rng.shuffle(files)
                for f in files[:cap]:
                    X.append(np.asarray(Image.open(f).convert("RGB").resize((S, S), Image.BILINEAR)))
                    Y.append(ci)
            log(f"  {split}: {len(X)} images")
    else:
        raise SystemExit("dataset inconnu")
    np.savez_compressed(OUT / f"data_{args.dataset}_{S}.npz", xtr=np.stack(xtr), ytr=np.array(ytr), xva=np.stack(xva),
                        yva=np.array(yva), classes=np.array(classes))
    log("donnees ecrites ->", OUT / f"data_{args.dataset}_{S}.npz", f"train={len(xtr)} val={len(xva)}")


class Data:
    def __init__(self, args):
        d = np.load(OUT / f"data_{args.dataset}_{args.img_size}.npz", allow_pickle=True)
        self.xtr = torch.from_numpy(d["xtr"]).permute(0, 3, 1, 2).contiguous()
        self.ytr = torch.from_numpy(d["ytr"]).long()
        self.xva = torch.from_numpy(d["xva"]).permute(0, 3, 1, 2).contiguous()
        self.yva = torch.from_numpy(d["yva"]).long()
        self.classes = [str(c) for c in d["classes"]]
        self.nc = len(self.classes)
        self.bs = args.batch_size

    @staticmethod
    def norm(x):
        return (x.float() / 255.0 - MEAN) / STD

    def train_batches(self, rng, augment=True):
        idx = torch.from_numpy(rng.permutation(len(self.xtr)))
        for i in range(0, len(idx) - self.bs + 1, self.bs):
            b = idx[i:i + self.bs]
            x = self.norm(self.xtr[b])
            if augment:
                flip = torch.rand(len(b)) < 0.5
                x[flip] = x[flip].flip(3)
                pad = max(2, x.shape[-1] // 16)
                x = F.pad(x, (pad,) * 4, mode="reflect")
                dx, dy = rng.randint(0, 2 * pad + 1, 2)
                x = x[:, :, dy:dy + self.xtr.shape[2], dx:dx + self.xtr.shape[3]]
            yield x, self.ytr[b]

    def val_batches(self, bs=64, n=None):
        N = len(self.xva) if n is None else min(n, len(self.xva))
        for i in range(0, N, bs):
            yield self.norm(self.xva[i:i + bs]), self.yva[i:i + bs]

    def calib(self, n=500):
        sel = torch.randperm(len(self.xtr))[:n]
        for i in range(0, len(sel), 32):
            yield self.norm(self.xtr[sel[i:i + 32]])


# ============================================================================ modele / mesures
def make_model(nc, pretrained=True):
    import torchvision
    try:
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet18(weights=w)
    except Exception as e:                                           # pas de reseau : poids aleatoires (declare)
        log("[ATTENTION] poids pre-entraines indisponibles :", type(e).__name__, "-> initialisation aleatoire")
        m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, nc)
    return m


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def size_mb(m):
    buf = io.BytesIO()
    torch.save(m.state_dict() if not hasattr(m, "_is_fx_q") else m.state_dict(), buf)
    return buf.tell() / 1e6


def macs(m, S):
    import torch_pruning as tp
    try:
        return tp.utils.count_ops_and_params(m.eval(), torch.randn(1, 3, S, S))[0] / 1e6
    except Exception:
        return float("nan")


@torch.no_grad()
def evaluate(m, data):
    m.eval()
    conf = np.zeros((data.nc, data.nc), dtype=np.int64)
    for x, y in data.val_batches():
        p = m(x).argmax(1)
        for a, b in zip(y.tolist(), p.tolist()):
            conf[a, b] += 1
    tp_ = np.diag(conf).astype(float)
    prec = tp_ / np.maximum(conf.sum(0), 1)
    rec = tp_ / np.maximum(conf.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return {"acc": 100 * tp_.sum() / conf.sum(), "precision": 100 * prec.mean(), "recall": 100 * rec.mean(), "f1": 100 * f1.mean()}


@torch.no_grad()
def latency_ms(m, S, threads=4, repeats=30):
    torch.set_num_threads(threads)
    m.eval()
    x = torch.randn(1, 3, S, S)
    for _ in range(5):
        m(x)
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        m(x)
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


def describe(m, data, S, threads, tag):
    r = evaluate(m, data)
    r.update({"params_M": n_params(m) / 1e6 if not hasattr(m, "_is_fx_q") else None, "size_mb": size_mb(m),
              "latency_ms": latency_ms(m, S, threads), "tag": tag})
    return r


# ============================================================================ entrainement
def train_ce(model, data, epochs, lr, rng, desc=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    for ep in range(epochs):
        model.train()
        tot = 0.0
        nb = 0
        for x, y in data.train_batches(rng):
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sch.step()
        if (ep + 1) % max(1, epochs // 4) == 0 or ep == epochs - 1:
            log(f"    {desc} CE ep {ep + 1}/{epochs} loss={tot / max(nb, 1):.3f} acc={evaluate(model, data)['acc']:.2f}")
    return model


def train_kd(student, teacher, data, epochs, lr, rng, T=4.0, alpha=0.3, desc=""):
    teacher.eval()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    for ep in range(epochs):
        student.train()
        tot = 0.0
        nb = 0
        for x, y in data.train_batches(rng):
            with torch.no_grad():
                tl = teacher(x)
            sl = student(x)
            loss = alpha * F.cross_entropy(sl, y) + (1 - alpha) * (T ** 2) * F.kl_div(
                F.log_softmax(sl / T, 1), F.softmax(tl / T, 1), reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sch.step()
        if (ep + 1) % max(1, epochs // 4) == 0 or ep == epochs - 1:
            log(f"    {desc} KD ep {ep + 1}/{epochs} loss={tot / max(nb, 1):.3f} acc={evaluate(student, data)['acc']:.2f}")
    return student


# ============================================================================ elagage structure (Torch-Pruning)
def conv_scores(model, criterion, data, n_batches=15):
    """Score par canal de sortie pour chaque Conv2d (dict nom -> tenseur)."""
    convs = {n: m for n, m in model.named_modules() if isinstance(m, nn.Conv2d)}
    if criterion in ("l1", "l2"):
        p = 1 if criterion == "l1" else 2
        return {n: m.weight.detach().abs().pow(p).flatten(1).sum(1) for n, m in convs.items()}
    acc = {n: torch.zeros(m.out_channels) for n, m in convs.items()}
    model.eval()
    rng = np.random.RandomState(0)
    if criterion == "activation":
        hooks = []
        for n, m in convs.items():
            hooks.append(m.register_forward_hook(lambda mod, i, o, n=n: acc.__setitem__(n, acc[n] + o.detach().abs().mean((0, 2, 3)))))
        with torch.no_grad():
            for b, (x, _) in enumerate(data.train_batches(rng, augment=False)):
                model(x)
                if b + 1 >= n_batches:
                    break
        for h in hooks:
            h.remove()
        return acc
    if criterion == "taylor":                                      # somme sur les lots de |w * dL/dw| (sans mise a jour)
        for b, (x, y) in enumerate(data.train_batches(rng, augment=False)):
            model.zero_grad()
            F.cross_entropy(model(x), y).backward()
            for n, m in convs.items():
                acc[n] += (m.weight.detach() * m.weight.grad).abs().flatten(1).sum(1)
            if b + 1 >= n_batches:
                break
        model.zero_grad()
        return acc
    raise ValueError(criterion)


def make_importance(scores, model):
    import torch_pruning as tp
    from torch_pruning.pruner import function
    name_of = {m: n for n, m in model.named_modules()}

    class ScoreImportance(tp.importance.GroupMagnitudeImportance):
        def __call__(self, group):
            gi, gx = [], []
            for i, (dep, idxs) in enumerate(group):
                layer = dep.layer
                if isinstance(layer, nn.Conv2d) and getattr(dep.pruning_fn, "__name__", "") in ("prune_out_channels", "prune_conv_out_channels"):
                    gi.append(scores[name_of[layer]][idxs].float())
                    gx.append(group[i].root_idxs)
            if not gi:
                return None
            s = self._reduce(gi, gx)
            lo = torch.quantile(s, 0.99) if len(s) > 10 else s.max()   # ecretage au 99e percentile puis min-max
            s = s.clamp(max=lo)
            return (s - s.min()) / (s.max() - s.min() + 1e-8)

    return ScoreImportance(group_reduction="sum", normalizer=None)


def prune_model(model, scores, target_reduction, S):
    """Elague par canaux (classement global) pour obtenir ~target_reduction % de parametres en moins (dichotomie)."""
    import torch_pruning as tp
    base = n_params(model)
    ex = torch.randn(1, 3, S, S)
    lo, hi = 0.0, 0.97
    best = None
    for _ in range(10):
        mid = (lo + hi) / 2
        m = copy.deepcopy(model).eval()
        imp = make_importance(scores, m)
        pr = tp.pruner.MetaPruner(m, ex, importance=imp, pruning_ratio=mid, global_pruning=True,
                                  ignored_layers=[m.fc], max_pruning_ratio=0.95)
        pr.step()
        red = 100 * (base - n_params(m)) / base
        best = (m, red, mid) if best is None or abs(red - target_reduction) < abs(best[1] - target_reduction) else best
        if red < target_reduction:
            lo = mid
        else:
            hi = mid
    m, red, ratio = best
    for p in m.parameters():
        p.requires_grad_(True)
    return m, red, ratio


def reinit(model):
    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(mod.weight, nonlinearity="relu")
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)


# ============================================================================ INT8 (FX statique)
def pick_backend():
    import torch.backends.quantized as q
    for b in ("fbgemm", "x86", "onednn", "qnnpack"):
        if b in q.supported_engines:
            return b
    raise RuntimeError("aucun backend de quantification")


def ptq_int8(model, data, n_calib=500):
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    backend = pick_backend()
    torch.backends.quantized.engine = backend
    m = copy.deepcopy(model).eval().cpu()
    ex = (torch.randn(1, 3, data.xtr.shape[2], data.xtr.shape[3]),)
    p = prepare_fx(m, get_default_qconfig_mapping(backend), example_inputs=ex)
    with torch.no_grad():
        for x in data.calib(n_calib):
            p(x)
    q = convert_fx(p)
    q._is_fx_q = True
    return q, backend


# ============================================================================ experiences
def get_baseline(args, data):
    sd = OUT / "r18_baseline.pt"
    m = make_model(data.nc, not args.no_pretrained)
    if sd.is_file():
        m.load_state_dict(torch.load(sd, map_location="cpu"))
        return m.eval()
    log(f"[baseline] fine-tuning ResNet-18 ({args.base_epochs} epoques)")
    train_ce(m, data, args.base_epochs, args.lr_base, np.random.RandomState(args.seed), "baseline")
    torch.save(m.state_dict(), sd)
    jdump(f"{args.dataset}_baseline", describe(m, data, args.img_size, args.threads, "baseline"))
    return m.eval()


def cmd_baseline(args):
    data = Data(args)
    torch.manual_seed(args.seed)
    get_baseline(args, data)


SPARS = [20, 50, 80, 90, 95]


def cmd_criteria(args):
    """5.2.2 : 4 criteres x niveaux de sparsite : elague seul / distille."""
    data = Data(args)
    torch.manual_seed(args.seed)
    base = get_baseline(args, data)
    for crit in CRITERIA:
        scores = conv_scores(copy.deepcopy(base), crit, data)
        for sp in args.sparsities:
            name = f"{args.dataset}_crit_{crit}_{sp}"
            if jload(name):
                continue
            log(f"[criteres] {crit} / reduction {sp}%")
            pm, red, ratio = prune_model(base, scores, sp, args.img_size)
            pruned = evaluate(pm, data)
            st = train_kd(pm, copy.deepcopy(base), data, args.kd_epochs, args.lr_kd, np.random.RandomState(args.seed), desc=f"{crit}{sp}")
            dist = evaluate(st, data)
            jdump(name, {"criterion": crit, "target": sp, "params_reduction_pct": red, "channel_ratio": ratio,
                         "params_M": n_params(st) / 1e6, "pruned": pruned, "distilled": dist})


def cmd_ablations(args):
    """5.2.3 (KD vs CE) et 5.2.4 (re-initialisation) : critere Taylor, sparsites 50/80/90."""
    data = Data(args)
    torch.manual_seed(args.seed)
    base = get_baseline(args, data)
    scores = conv_scores(copy.deepcopy(base), "taylor", data)
    for sp in (50, 80, 90):
        name = f"{args.dataset}_abl_{sp}"
        if jload(name):
            continue
        log(f"[ablations] reduction {sp}%")
        out = {"target": sp}
        for variant in ("kd_keep", "ce_keep", "kd_reinit", "ce_reinit"):
            pm, red, _ = prune_model(base, scores, sp, args.img_size)
            if variant.endswith("reinit"):
                reinit(pm)
            rng = np.random.RandomState(args.seed)
            if variant.startswith("kd"):
                train_kd(pm, copy.deepcopy(base), data, args.kd_epochs, args.lr_kd, rng, desc=variant)
            else:
                train_ce(pm, data, args.kd_epochs, args.lr_kd, rng, variant)
            out[variant] = evaluate(pm, data)
            out["params_reduction_pct"] = red
        jdump(name, out)


def cmd_int8(args):
    """5.2.5 : INT8 statique seul sur la baseline."""
    data = Data(args)
    torch.manual_seed(args.seed)
    base = get_baseline(args, data)
    q, backend = ptq_int8(base, data)
    fp = describe(base, data, args.img_size, args.threads, "fp32")
    i8 = describe(q, data, args.img_size, args.threads, "int8")
    jdump(f"{args.dataset}_int8_standalone", {"backend": backend, "fp32": fp, "int8": i8})


def cmd_pipeline(args):
    """5.2.6 : baseline -> elagage Taylor -> distillation -> INT8."""
    data = Data(args)
    torch.manual_seed(args.seed)
    base = get_baseline(args, data)
    S = args.img_size
    stages = [dict(describe(base, data, S, args.threads, "Stage 0 FP32 baseline"), macs_M=macs(base, S))]
    scores = conv_scores(copy.deepcopy(base), "taylor", data)
    pm, red, _ = prune_model(base, scores, args.pipeline_reduction, S)
    stages.append(dict(describe(pm, data, S, args.threads, "Stage 1 structured pruned"), macs_M=macs(pm, S)))
    train_kd(pm, copy.deepcopy(base), data, args.pipe_epochs, args.lr_kd, np.random.RandomState(args.seed), desc="pipeline")
    stages.append(dict(describe(pm, data, S, args.threads, "Stage 2 KD recovery"), macs_M=macs(pm, S)))
    q, backend = ptq_int8(pm, data)
    d = describe(q, data, S, args.threads, "Stage 3 INT8 PTQ")
    d["params_M"] = stages[2]["params_M"]
    d["macs_M"] = stages[2]["macs_M"]
    stages.append(d)
    jdump(f"{args.dataset}_pipeline", {"backend": backend, "params_reduction_pct": red, "stages": stages})
    for s in stages:
        log(f"  {s['tag']:28} acc={s['acc']:.2f} f1={s['f1']:.2f} size={s['size_mb']:.2f}MB lat={s['latency_ms']:.1f}ms")


def cmd_all(args):
    cmd_baseline(args)
    cmd_pipeline(args)
    cmd_int8(args)
    cmd_criteria(args)
    cmd_ablations(args)
    cmd_report(args)


# ============================================================================ rapport (figures + tex)
def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                         "figure.dpi": 150, "savefig.bbox": "tight"})
    ds = args.dataset
    COL = {"l1": "#5e81ac", "l2": "#bf616a", "activation": "#a3be8c", "taylor": "#d08770"}
    tex = []

    # ---- criteres
    crit = {c: [jload(f"{ds}_crit_{c}_{s}") for s in SPARS] for c in CRITERIA}
    if any(r for v in crit.values() for r in v):
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
        for ax, c in zip(axes, CRITERIA):
            rs = [r for r in crit[c] if r]
            xs = [r["target"] for r in rs]
            ax.plot(xs, [r["pruned"]["acc"] for r in rs], "o--", color="#4c566a", label="pruned only")
            ax.plot(xs, [r["distilled"]["acc"] for r in rs], "o-", color=COL[c], label="pruned + KD")
            ax.set_title(f"{c.upper()} pruning")
            ax.set_xlabel("parameter reduction (%)")
        axes[0].set_ylabel("accuracy (%)")
        axes[0].legend(frameon=False)
        fig.savefig(FIG / "r18_criteria_each.png")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for c in CRITERIA:
            rs = [r for r in crit[c] if r]
            ax.plot([r["target"] for r in rs], [r["distilled"]["f1"] for r in rs], "o-", color=COL[c], label=c)
        ax.set_xlabel("parameter reduction (%)")
        ax.set_ylabel("macro-F1 after KD (%)")
        ax.set_title("Distilled F1 for the four pruning criteria")
        ax.legend(frameon=False)
        fig.savefig(FIG / "r18_criteria_f1.png")
        plt.close(fig)
        rows = ["\\begin{tabular}{l" + "c" * len(SPARS) + "}\\toprule", "Criterion & " + " & ".join(f"{s}\\%" for s in SPARS) + "\\\\\\midrule"]
        for c in CRITERIA:
            rows.append(c.upper() + " & " + " & ".join(f"{r['distilled']['f1']:.1f}" if r else "--" for r in crit[c]) + "\\\\")
        rows.append("\\bottomrule\\end{tabular}")
        (TAB / "r18_criteria_f1.tex").write_text("\n".join(rows), encoding="utf-8")
        tex.append("criteria")

    # ---- ablations
    ab = [jload(f"{ds}_abl_{s}") for s in (50, 80, 90)]
    ab = [a for a in ab if a]
    if ab:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        w = 0.2
        for j, (v, lab) in enumerate([("ce_keep", "CE, weights kept"), ("kd_keep", "KD, weights kept"),
                                      ("ce_reinit", "CE, Kaiming re-init"), ("kd_reinit", "KD, Kaiming re-init")]):
            ax.bar(np.arange(len(ab)) + (j - 1.5) * w, [a[v]["acc"] for a in ab], w, label=lab)
        ax.set_xticks(range(len(ab)))
        ax.set_xticklabels([f"{a['target']}%" for a in ab])
        ax.set_xlabel("parameter reduction")
        ax.set_ylabel("accuracy (%)")
        ax.set_title("Distillation vs. cross-entropy recovery, and effect of re-initialisation (Taylor)")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(FIG / "r18_ablations.png")
        plt.close(fig)
        rows = ["\\begin{tabular}{lcccc}\\toprule", "Reduction & CE keep & KD keep & CE re-init & KD re-init\\\\\\midrule"]
        for a in ab:
            rows.append(f"{a['target']}\\% & " + " & ".join(f"{a[v]['acc']:.1f}" for v in ("ce_keep", "kd_keep", "ce_reinit", "kd_reinit")) + "\\\\")
        rows.append("\\bottomrule\\end{tabular}")
        (TAB / "r18_ablations.tex").write_text("\n".join(rows), encoding="utf-8")

    # ---- INT8 seul + pipeline
    i8 = jload(f"{ds}_int8_standalone")
    if i8:
        f, q = i8["fp32"], i8["int8"]
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, (k, t) in zip(axes, [("acc", "accuracy (%)"), ("size_mb", "size (MB)"), ("latency_ms", "latency (ms)")]):
            ax.bar(["FP32", "INT8"], [f[k], q[k]], color=["#5e81ac", "#d08770"])
            ax.set_title(t)
        fig.suptitle(f"Standalone INT8 PTQ (backend {i8['backend']})", y=1.08)
        fig.savefig(FIG / "r18_int8_standalone.png")
        plt.close(fig)
        rows = ["\\begin{tabular}{lccc}\\toprule", "Metric & FP32 & INT8 & $\\Delta$\\\\\\midrule"]
        for k, lab in [("acc", "Accuracy (\\%)"), ("f1", "Macro-F1 (\\%)"), ("size_mb", "Size (MB)"), ("latency_ms", "Latency (ms)")]:
            rows.append(f"{lab} & {f[k]:.2f} & {q[k]:.2f} & {q[k] - f[k]:+.2f}\\\\")
        rows.append("\\bottomrule\\end{tabular}")
        (TAB / "r18_int8_standalone.tex").write_text("\n".join(rows), encoding="utf-8")
    pp = jload(f"{ds}_pipeline")
    if pp:
        st = pp["stages"]
        names = ["S0\nFP32", "S1\npruned", "S2\nKD", "S3\nINT8"]
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
        for ax, (k, t) in zip(axes, [("acc", "accuracy (%)"), ("f1", "macro-F1 (%)"), ("size_mb", "size (MB)"), ("params_M", "parameters (M)")]):
            ax.bar(names, [s[k] for s in st], color=["#5e81ac", "#bf616a", "#a3be8c", "#d08770"])
            ax.set_title(t)
        fig.suptitle(f"Full P-KD-Q pipeline on {ds}", y=1.08)
        fig.savefig(FIG / "r18_pipeline.png")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.bar(names, [s["latency_ms"] for s in st], color=["#5e81ac", "#bf616a", "#a3be8c", "#d08770"])
        ax.set_ylabel("CPU latency (ms)")
        ax.set_title("Latency across the four stages")
        fig.savefig(FIG / "r18_pipeline_latency.png")
        plt.close(fig)
        rows = ["\\begin{tabular}{lrrrrrr}\\toprule", "Stage & Acc (\\%) & F1 (\\%) & Size (MB) & Params (M) & MACs (M) & Lat (ms)\\\\\\midrule"]
        for s in st:
            rows.append(f"{s['tag']} & {s['acc']:.2f} & {s['f1']:.2f} & {s['size_mb']:.2f} & {s['params_M']:.2f} & {s.get('macs_M', float('nan')):.0f} & {s['latency_ms']:.1f}\\\\")
        rows.append("\\bottomrule\\end{tabular}")
        (TAB / "r18_pipeline.tex").write_text("\n".join(rows), encoding="utf-8")
        s0, s1, s2, s3 = st
        ratio = s0["size_mb"] / s3["size_mb"]
        (TAB / "r18_pipeline_summary.txt").write_text(
            f"compression {ratio:.1f}x ({s0['size_mb']:.2f}->{s3['size_mb']:.2f} MB); accuracy {s0['acc']:.2f}->{s3['acc']:.2f} "
            f"({s3['acc'] - s0['acc']:+.2f}); pruned-only {s1['acc']:.2f}; KD {s2['acc']:.2f}; INT8 delta {s3['acc'] - s2['acc']:+.2f}; "
            f"latency {s0['latency_ms']:.1f}->{s3['latency_ms']:.1f} ms\n", encoding="utf-8")
    log("figures ->", FIG, "| tableaux ->", TAB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "baseline", "criteria", "ablations", "int8", "pipeline", "report", "all"])
    ap.add_argument("--dataset", default="coco7", choices=["coco7", "imagenette", "synthetic"])
    ap.add_argument("--coco-images", default=None)
    ap.add_argument("--coco-ann", default=None)
    ap.add_argument("--min-box", type=int, default=32)
    ap.add_argument("--max-train", type=int, default=800)
    ap.add_argument("--max-val", type=int, default=250)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--base-epochs", type=int, default=8)
    ap.add_argument("--kd-epochs", type=int, default=6, help="epoques de recuperation (criteres / ablations)")
    ap.add_argument("--pipe-epochs", type=int, default=15, help="epoques de recuperation du pipeline complet")
    ap.add_argument("--lr-base", type=float, default=3e-4)
    ap.add_argument("--lr-kd", type=float, default=1e-3)
    ap.add_argument("--pipeline-reduction", type=float, default=81.0, help="% de parametres retires (rapport : 11.18M -> 2.08M)")
    ap.add_argument("--sparsities", type=int, nargs="+", default=SPARS)
    args = ap.parse_args()
    global OUT, RES, FIG, TAB
    torch.set_num_threads(args.threads)
    if args.cmd != "prepare":
        OUT = Path("r18_report") / args.dataset if False else OUT
    {"prepare": prepare, "baseline": cmd_baseline, "criteria": cmd_criteria, "ablations": cmd_ablations, "int8": cmd_int8,
     "pipeline": cmd_pipeline, "report": cmd_report, "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()