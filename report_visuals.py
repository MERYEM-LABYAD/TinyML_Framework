"""
Figures du rapport (courbes, histogrammes, comparaisons, exemples de detections).

Depuis E:\\Stage2\\TinyML_Framework (verify_m0_8000.py et evaluate_m0_m1_pruned.py dans le meme dossier) :

  1) Figures chiffrees (RAPIDE, pas de modele, quelques secondes) :
       python report_visuals.py figures
     Lit automatiquement les eval.json de m1_report\\checkpoints\\*_eval\\ (M1, M2, M3 de chaque config) et
     les courbes de perte m1_report\\m3_losses_*.csv. Ce qui n'existe pas encore est simplement saute
     (ou pris dans les valeurs de reference ci-dessous). Sortie : m1_report\\figures\\*.png

  2) Tableaux LaTeX auto-generes (conception des experiences, couts) :
       python report_visuals.py tables          -> m1_report\\tables\\design.tex et costs.tex

  3) Exemples qualitatifs, style figure 34 du rapport (LENT : ~9 s/image/modele sur CPU) :
       python report_visuals.py examples --config expose_whole_backbone --n 4
       python report_visuals.py examples --config aggressive --n 4
     UN isolateur par ligne ; colonnes M0 | M1 | M2 | M3 ; verite terrain en VERT, prediction en ROUGE.
     Par defaut on choisit des isolateurs que M0 detecte correctement (cas representatifs) ; --any pour tout
     tirer au hasard ; --classes "Broken,Flashover damage" pour ne montrer que des defauts.
     Sortie : m1_report\\figures\\examples_<config>.png
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("m1_report")
CKPT = ROOT / "checkpoints"
FIG = ROOT / "figures"

M0 = {"AP": 71.19, "AP50": 92.15, "AP75": 85.51, "parameters": 41305311, "size_mb": 165.6}

# Configs : nom court -> (nom de fichier checkpoint sans prefixe stage, libelle)
CONFIGS = {
    "expose_whole_backbone": ("expose_whole_backbone_t41.8_crank", "Whole backbone (-41.8 %)"),
    "aggressive": ("aggressive_t57.6_crank", "Aggressive (-57.6 %)"),
    "maximal": ("maximal_t61.7_crank", "Maximal (-61.7 %)"),
}

# Valeurs deja mesurees (reprises de tes evaluations) ; ecrasees par les eval.json trouves sur le disque.
REF = {
    ("expose_whole_backbone", "M1"): {"AP": 0.4746, "AP50": None, "AP75": None, "retention_AP": 0.006667},
    ("expose_whole_backbone", "M2"): {"AP": 27.9, "retention_AP": 0.39205},
    ("expose_whole_backbone", "M3"): {"AP": 65.98, "AP50": 86.95, "AP75": 78.89, "Precision": 0.7915,
                                       "Recall": 0.9105, "F1": 0.8469, "retention_AP": 0.92687,
                                       "parameters": 24026537, "size_mb_state_dict": 96.35},
    ("aggressive", "M2"): {"AP": 1.768, "AP50": 5.547, "AP75": 0.529, "retention_AP": 0.02484},
    ("aggressive", "M3"): {"AP": 48.77, "AP50": 67.96, "AP75": 58.94, "Precision": 0.7130, "Recall": 0.8250,
                            "F1": 0.7649, "retention_AP": 0.68505,
                            "parameters": 17507472, "size_mb_state_dict": 70.18},
    ("maximal", "M2"): {"AP": 0.366, "AP50": 1.62, "retention_AP": 0.005141},
    ("maximal", "M3"): {"AP": 46.38, "AP50": 65.19, "AP75": 56.02, "Precision": 0.7341, "Recall": 0.8214,
                         "F1": 0.7753, "retention_AP": 0.65157,
                         "parameters": 15808632, "size_mb_state_dict": 63.34},
}
STAGES = ["M0", "M1", "M2", "M3"]
COL = {"M0": "#4c566a", "M1": "#bf616a", "M2": "#d08770", "M3": "#5e81ac"}


def load_eval(config: str, stage: str):
    """eval.json du disque si present, sinon valeur de reference, sinon None."""
    base = CONFIGS[config][0]
    folder = {"M1": f"M1R_{base}_eval", "M2": f"M2_{base}_matched_eval", "M3": f"M3_{base}_matched_eval"}[stage]
    p = CKPT / folder / "eval.json"
    d = dict(REF.get((config, stage), {}))
    if p.is_file():
        d.update(json.loads(p.read_text(encoding="utf-8")))
    return d or None


def coco_results(config: str, stage: str):
    base = CONFIGS[config][0]
    folder = {"M1": f"M1R_{base}_eval", "M2": f"M2_{base}_matched_eval", "M3": f"M3_{base}_matched_eval"}[stage]
    p = CKPT / folder / "coco_instances_results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def retention(d):
    if d is None:
        return None
    if d.get("retention_AP") is not None:
        return 100 * d["retention_AP"]
    return 100 * d["AP"] / M0["AP"] if d.get("AP") is not None else None


def style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150, "savefig.bbox": "tight"})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print("  ->", FIG / name)


# ----------------------------------------------------------------------------- figures chiffrees
def fig_retention():
    fig, ax = plt.subplots(figsize=(7, 4))
    for cfg, (_, lab) in CONFIGS.items():
        ys = [100.0] + [retention(load_eval(cfg, s)) for s in STAGES[1:]]
        xs = [i for i, y in enumerate(ys) if y is not None]
        yv = [y for y in ys if y is not None]
        ax.plot(xs, yv, marker="o", label=lab)
        for x, y in zip(xs, yv):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["M0\nteacher", "M1\npruning", "M2\nrecalibration", "M3\ndistillation"])
    ax.set_ylabel("AP retention (% of M0)")
    ax.set_title("AP recovery stage by stage")
    ax.legend(frameon=False)
    save(fig, "retention_per_stage.png")


def fig_ap_bars():
    cfgs = list(CONFIGS)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, key in zip(axes, ["AP", "AP50", "AP75"]):
        w = 0.2
        for j, s in enumerate(STAGES):
            vals = []
            for c in cfgs:
                d = M0 if s == "M0" else load_eval(c, s)
                vals.append(np.nan if d is None or d.get(key) is None else d[key])
            ax.bar(np.arange(len(cfgs)) + (j - 1.5) * w, vals, w, label=s, color=COL[s])
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels(["Whole", "Aggressive", "Maximal"])
        ax.set_title(key)
    axes[0].set_ylabel("AP points")
    axes[0].legend(frameon=False, ncol=4, loc="upper right")
    fig.suptitle("COCO metrics per model version (test set)", y=1.02)
    save(fig, "ap_per_version.png")


def fig_size():
    labs, par, mb = ["M0"], [M0["parameters"] / 1e6], [M0["size_mb"]]
    for c, (_, lab) in CONFIGS.items():
        d = load_eval(c, "M3")
        if d and d.get("parameters"):
            labs.append(lab.split(" (")[0])
            par.append(d["parameters"] / 1e6)
            mb.append(d["size_mb_state_dict"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.5))
    a.bar(labs, par, color=["#4c566a"] + ["#5e81ac"] * (len(labs) - 1))
    a.set_ylabel("Parameters (millions)")
    b.bar(labs, mb, color=["#4c566a"] + ["#5e81ac"] * (len(labs) - 1))
    b.set_ylabel("state_dict size (MB)")
    for ax, v in ((a, par), (b, mb)):
        for i, y in enumerate(v):
            ax.text(i, y, f"{y:.1f}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", labelrotation=15)
    fig.suptitle("Achieved compression")
    save(fig, "size_params.png")


def fig_prf():
    rows = [(lab.split(" (")[0], load_eval(c, "M3")) for c, (_, lab) in CONFIGS.items()]
    rows = [(l, d) for l, d in rows if d and d.get("Precision") is not None]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    w = 0.25
    for j, k in enumerate(["Precision", "Recall", "F1"]):
        ax.bar(np.arange(len(rows)) + (j - 1) * w, [d[k] for _, d in rows], w, label=k)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([l for l, _ in rows])
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=3)
    ax.set_title("Precision / recall / F1 (score 0.5, IoU 0.5) after M3")
    save(fig, "prf.png")


def fig_per_class():
    names, data = None, {}
    for c in CONFIGS:
        d = load_eval(c, "M3")
        if d and d.get("AP_per_class"):
            data[c] = d["AP_per_class"]
    if not data:
        return
    names = list(next(iter(data.values())).keys())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.25
    for j, (c, v) in enumerate(data.items()):
        ax.bar(np.arange(len(names)) + j * w, [v[n] for n in names], w, label=CONFIGS[c][1])
    ax.set_xticks(np.arange(len(names)) + w / 2)
    ax.set_xticklabels([n.replace("AP-", "") for n in names])
    ax.set_ylabel("Per-class AP")
    ax.legend(frameon=False)
    ax.set_title("Per-class AP after M3")
    save(fig, "ap_per_class.png")


def fig_score_hist():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
    any_data = False
    for ax, (c, (_, lab)) in zip(axes, CONFIGS.items()):
        for s in ["M2", "M3"]:
            r = coco_results(c, s)
            if r:
                any_data = True
                ax.hist([x["score"] for x in r], bins=20, range=(0, 1), alpha=0.6, label=s, color=COL[s])
        ax.set_title(lab)
        ax.set_xlabel("confidence score")
        ax.legend(frameon=False)
    axes[0].set_ylabel("number of detections")
    fig.suptitle("Detection score distribution (threshold 0.05)", y=1.02)
    if any_data:
        save(fig, "score_histogram.png")
    else:
        plt.close(fig)


def fig_loss():
    files = sorted(ROOT.glob("m3_losses_*.csv"))
    if not files:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    for f in files:
        rows = list(csv.DictReader(f.open(encoding="utf-8")))
        if not rows:
            continue
        it = np.array([float(r.get("iter", i + 1)) for i, r in enumerate(rows)])
        for ax, key in ((a, "task" if "task" in rows[0] else "loss"), (b, "feat_mse" if "feat_mse" in rows[0] else "loss")):
            y = np.array([float(r[key]) for r in rows])
            k = max(1, min(20, len(y) // 8))
            sm = np.convolve(y, np.ones(k) / k, mode="valid") if len(y) > k else y
            ax.plot(it[len(it) - len(sm):], sm, label=f.stem.replace("m3_losses_", "").split("_t")[0])
    a.set_title("Detection loss (RPN + ROI)")
    b.set_title("Distillation loss (FPN P2-P5)")
    for ax in (a, b):
        ax.set_xlabel("iteration")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("M3 training curves (20-iteration moving average)", y=1.02)
    save(fig, "m3_loss_curves.png")



def fig_channels():
    """Canaux supprimes par bloc (equivalent de la figure 'Channels removed per block' du rapport)."""
    import re
    cfgs = []
    for c, (base, lab) in CONFIGS.items():
        p = CKPT / f"M1R_{base}.report.json"
        if p.is_file():
            cfgs.append((c, lab, json.loads(p.read_text(encoding="utf-8"))))
    if not cfgs:
        print("  [SKIP] fig_channels : aucun M1R_*.report.json trouve")
        return
    fig, axes = plt.subplots(len(cfgs), 1, figsize=(11, 3.1 * len(cfgs)), squeeze=False)
    for ax, (c, lab, rep) in zip(axes[:, 0], cfgs):
        blocks = [("stem", "stem")]
        for st, n in (("res2", 3), ("res3", 4), ("res4", 6), ("res5", 3)):
            blocks += [(f"{st}.{i}", f"{st}.{i}") for i in range(n)]
        pct = {k: {"conv1": np.nan, "conv2": np.nan, "res": np.nan} for k, _ in blocks}
        for g in rep["channel_map"]:
            for m in g["members"]:
                mm = re.search(r"bottom_up\.(res\d\.\d)\.(conv1|conv2)$", m)
                if mm and g["label"] == "internal":
                    pct[mm.group(1)][mm.group(2)] = g["removed_pct"]
                mm = re.search(r"bottom_up\.(res\d\.\d)\.conv3$", m)
                if mm and g["label"] == "residual":
                    pct[mm.group(1)]["res"] = g["removed_pct"]
                if m.endswith("stem.conv1"):
                    pct["stem"]["conv1"] = g["removed_pct"]
        x = np.arange(len(blocks))
        w = 0.27
        for j, (k, nm, col) in enumerate((("conv1", "conv1 (interne)", "#5e81ac"), ("conv2", "conv2 (interne)", "#88c0d0"),
                                          ("res", "conv3 / sortie residuelle (partagee par etage)", "#d08770"))):
            ax.bar(x + (j - 1) * w, [pct[b][k] for b, _ in blocks], w, label=nm, color=col)
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in blocks], rotation=60, fontsize=7)
        ax.set_ylabel("channels removed (%)")
        ref = rep.get("report_reference", {})
        ax.set_title(f"{lab}  -  reference: {ref.get('ref_text', '')}", fontsize=9)
        ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper right")
        ax.set_ylim(0, 100)
    fig.suptitle("Channels removed per ResNet-50 backbone block", y=1.0)
    fig.tight_layout()
    save(fig, "channels_per_block.png")


# points du rapport de reference (reduction de parametres %, retention AP %)
REPORT_POINTS = {"Whole backbone": (41.8, 95.7), "Aggressive": (57.6, 88.9), "Maximal": (61.7, 81.2)}


def fig_pareto():
    """Compromis taille / precision : nos points face a ceux du rapport (valeur ajoutee)."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(*zip(*REPORT_POINTS.values()), "s--", color="#4c566a", label="Reference report")
    for n, (x, y) in REPORT_POINTS.items():
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(4, 6), fontsize=7, color="#4c566a")
    xs, ys = [], []
    for c, (_, lab) in CONFIGS.items():
        d = load_eval(c, "M3")
        r = retention(d)
        if d and r is not None and d.get("parameters"):
            xs.append(100 * (M0["parameters"] - d["parameters"]) / M0["parameters"])
            ys.append(r)
    if xs:
        ax.plot(xs, ys, "o-", color="#5e81ac", label="Our reproduction (M3)")
    ax.set_xlabel("Parameter reduction (%)")
    ax.set_ylabel("AP retention (%)")
    ax.set_title("Compression / accuracy trade-off")
    ax.legend(frameon=False)
    save(fig, "pareto_tradeoff.png")


# AP vs taille / latence (valeurs mesurees par measure_costs.py / run_m4_report.py, reprises ici par defaut)
TRADE = {  # label: (AP %, taille MB, latence ms) ; latences = costs.json (session de mesure unique, 20 repetitions)
    "M0": (71.19, 165.6, 8126), "Whole (M3)": (65.98, 96.4, 5842), "Aggressive (M3)": (48.77, 70.2, 5402),
    "Maximal (M3)": (46.38, 63.3, 4786), "Maximal INT8 (M4)": (42.49, 18.2, None)}  # INT8 : latence mesuree dans une autre session -> non tracee


def fig_tradeoff():
    for idx, (xl, name, key) in enumerate([("Model size (MB)", "tradeoff_size.png", 1), ("CPU latency (ms, batch 1, 4 threads)", "tradeoff_latency.png", 2)]):
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        for lab, v in TRADE.items():
            if v[key] is None:
                continue
            ax.scatter(v[key], v[0], s=45)
            ax.annotate(lab, (v[key], v[0]), textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel(xl)
        ax.set_ylabel("AP50:95 (%)")
        ax.set_title("AP vs " + ("model size" if key == 1 else "CPU latency"))
        save(fig, name)



def fig_ap_size():
    fig, ax = plt.subplots(figsize=(7, 3.6))
    keys = [("APs", "petits"), ("APm", "moyens"), ("APl", "grands")]
    items = [("M0", json.loads((Path("E:/Stage2/Faster-rcnn/VERIFY_M0_8000/verify_m0_8000.json")).read_text(encoding="utf-8"))
              if Path("E:/Stage2/Faster-rcnn/VERIFY_M0_8000/verify_m0_8000.json").is_file() else None)]
    for c, (_, lab) in CONFIGS.items():
        items.append((lab.split(" (")[0], load_eval(c, "M3")))
    items = [(n, d) for n, d in items if d and d.get("APl") is not None]
    if len(items) < 2:
        plt.close(fig)
        return
    w = 0.8 / len(items)
    for j, (n, d) in enumerate(items):
        ax.bar(np.arange(3) + j * w, [d[k] for k, _ in keys], w, label=n)
    ax.set_xticks(np.arange(3) + 0.4 - w / 2)
    ax.set_xticklabels([l for _, l in keys])
    ax.set_ylabel("AP")
    ax.set_title("AP by object size")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "ap_by_size.png")


def cmd_figures(_):
    style()
    for fn in (fig_retention, fig_ap_bars, fig_size, fig_prf, fig_per_class, fig_score_hist, fig_loss,
               fig_channels, fig_pareto, fig_ap_size, fig_tradeoff):
        try:
            fn()
        except Exception as e:                       # une figure ne doit pas bloquer les autres
            print(f"  [SKIP] {fn.__name__} : {type(e).__name__}: {e}")


# ----------------------------------------------------------------------------- tableaux LaTeX auto-generes
TAB = ROOT / "tables"


def _tex(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def table_design():
    """Tableau de conception des experiences (equivalent du tableau 'ce que l'on va faire' du rapport)."""
    rows = []
    for c, (base, lab) in CONFIGS.items():
        p = CKPT / f"M1R_{base}.report.json"
        if not p.is_file():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        pcts = [v["removed_pct"] for k, v in r["summary_by_label"].items()
                if k.startswith(("internal", "residual"))]
        w = r.get("weights", {})
        wtxt = "/".join(f"{w[k]:g}" for k in ("stem", "internal", "residual", "rpn", "fc") if k in w)
        rows.append((lab.split(" (")[0], f"{r['params_before']:,}".replace(",", r"\,"),
                     f"{r['params_after']:,}".replace(",", r"\,"), f"{r['reduction_pct']:.1f}",
                     wtxt, f"{min(pcts):.0f}--{max(pcts):.0f}", _tex(r.get("report_reference", {}).get("ref_text", ""))))
    if not rows:
        return
    TAB.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{tabular}{@{}lrrrcll@{}}", r"\toprule",
             r"Config. & Params avant & Params apr\`es & R\'eduction (\%) & Poids (stem/int./res./RPN/fc) & "
             r"Canaux supprim\'es backbone (\%) & Plage du rapport \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "design.tex").write_text("\n".join(lines), encoding="utf-8")
    print("  ->", TAB / "design.tex")


def table_costs():
    p = ROOT / "costs.json"
    if not p.is_file():
        return
    rows = json.loads(p.read_text(encoding="utf-8"))["rows"]
    # ligne INT8 eventuelle (M4)
    for q in sorted(CKPT.glob("M4_*.quant.json")):
        j = json.loads(q.read_text(encoding="utf-8"))
        r = dict(j["int8"])
        r["label"] = "M4 INT8 (" + q.stem.replace("M4_", "").split("_t")[0] + ")"
        r["parameters"] = j["int8"]["parameters_float_equivalent"]
        rows.append(r)
    b = rows[0]
    TAB.mkdir(parents=True, exist_ok=True)
    L = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"Mod\`ele & Params (M) & Taille (Mo) & GFLOPs & $\Delta$GFLOPs & Latence (ms) & $\Delta$latence \\", r"\midrule"]
    for r in rows:
        gf = r.get("GFLOPs")
        dg = f"{100 * (gf / b['GFLOPs'] - 1):+.1f}\\%" if gf and b.get("GFLOPs") else "--"
        dl = f"{100 * (r['latency']['median_ms'] / b['latency']['median_ms'] - 1):+.1f}\\%"
        L.append(f"{_tex(r['label'])} & {r['parameters'] / 1e6:.2f} & {r.get('size_mb_state_dict', float('nan')):.1f} & "
                 f"{(f'{gf:.1f}' if gf else '--')} & {dg} & {r['latency']['median_ms']:.0f} & {dl} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "costs.tex").write_text("\n".join(L), encoding="utf-8")
    print("  ->", TAB / "costs.tex")


def cmd_tables(_):
    for fn in (table_design, table_costs):
        try:
            fn()
        except Exception as e:
            print(f"  [SKIP] {fn.__name__} : {type(e).__name__}: {e}")


# ----------------------------------------------------------------------------- exemples qualitatifs
# Equivalent de la figure 34 du reference: UN isolateur, verite terrain en VERT, prediction en ROUGE.
def preprocess(bgr, cfg):
    """Meme pretraitement que l'evaluation (DefaultPredictor) : plus petit cote -> 800 px (max 1333)."""
    import torch
    from detectron2.data import transforms as T
    aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
    img = aug.get_transform(bgr).apply_image(bgr)
    return {"image": torch.as_tensor(img.astype("float32").transpose(2, 0, 1)),
            "height": bgr.shape[0], "width": bgr.shape[1]}


def iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0.0, min(ax2, bx2) - max(a[0], b[0])), max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def best_pred(preds, gt_box, min_iou=0.1):
    """La prediction qui recouvre le mieux CET isolateur (None s'il n'y en a pas)."""
    best, bi = None, min_iou
    for p in preds:
        v = iou_xywh(p["bbox"], gt_box)
        if v > bi:
            best, bi = {**p, "iou": v}, v
    return best


def crop_box(gt_box, shape, margin=0.45):
    x, y, w, h = gt_box
    mx, my = margin * w, margin * h
    x1, y1 = max(0, int(x - mx)), max(0, int(y - my))
    x2, y2 = min(shape[1], int(x + w + mx)), min(shape[0], int(y + h + my))
    return x1, y1, x2, y2


def draw_crop(ax, rgb, window, gt_box, pred, title, class_names):
    import matplotlib.patches as mp
    x1, y1, x2, y2 = window
    ax.imshow(rgb[y1:y2, x1:x2])
    gx, gy, gw, gh = gt_box
    ax.add_patch(mp.Rectangle((gx - x1, gy - y1), gw, gh, fill=False, ec="#00c000", lw=2.2))
    if pred is not None:
        px, py, pw, ph = pred["bbox"]
        ax.add_patch(mp.Rectangle((px - x1, py - y1), pw, ph, fill=False, ec="#e00000", lw=2.2, ls="-"))
        ax.text(0.02, 0.97, f"{class_names[pred['category_id']]}  {pred['score']:.2f}", transform=ax.transAxes,
                va="top", color="white", fontsize=8, bbox=dict(fc="#e00000", ec="none", pad=1.5))
    else:
        ax.text(0.5, 0.5, "no detection", transform=ax.transAxes, ha="center", va="center",
                color="white", fontsize=9, bbox=dict(fc="#444444", ec="none", pad=2))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def run_model(model, cfg, bgr, thresh=0.5):
    import torch
    with torch.no_grad():
        out = model([preprocess(bgr, cfg)])[0]["instances"].to("cpu")
    preds = []
    for i in range(len(out)):
        s = float(out.scores[i])
        if s >= thresh:
            x1, y1, x2, y2 = out.pred_boxes.tensor[i].tolist()
            preds.append({"bbox": [x1, y1, x2 - x1, y2 - y1], "category_id": int(out.pred_classes[i]), "score": s})
    return preds


def load_models(ckpts, V):
    import torch
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from evaluate_m0_m1_pruned import reconstruct_pruned_architecture
    models = {}
    for k, p in ckpts.items():
        cfg = V.build_cfg(Path(p))
        m = build_model(cfg)
        if k == "M0":
            DetectionCheckpointer(m).load(str(p))
        else:
            sd = torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"]
            reconstruct_pruned_architecture(m, sd)
            m.load_state_dict(sd, strict=True)
        m.eval()
        models[k] = (m, cfg)
        print(f"[charge] {k}", flush=True)
    return models


def cmd_examples(args):
    import torch
    import cv2
    import verify_m0_8000 as V
    from detectron2.data import DatasetCatalog

    torch.set_num_threads(args.threads)
    style()
    base = CONFIGS[args.config][0]
    ckpts = {"M0": V.DEFAULT_CHECKPOINT, "M1": CKPT / f"M1R_{base}.pth",
             "M2": CKPT / f"M2_{base}_matched.pth", "M3": CKPT / f"M3_{base}_matched.pth"}
    ckpts = {k: p for k, p in ckpts.items() if Path(p).is_file()}
    for k in args.skip.split(","):
        ckpts.pop(k, None)

    V.register()
    gt = V.ground_truth_by_image()
    data = [d for d in DatasetCatalog.get(V.DATASET) if gt.get(d["image_id"])]
    random.Random(args.seed).shuffle(data)
    models = load_models(ckpts, V)
    m0_model, m0_cfg = models["M0"]

    chosen = []          # (image dict, index GT, predictions par modele)
    for d in data:
        bgr = cv2.imread(d["file_name"])
        p0 = run_model(m0_model, m0_cfg, bgr)
        # un isolateur que M0 detecte correctement (meme classe, IoU >= 0.5) : cas representatif
        for gi, g in enumerate(gt[d["image_id"]]):
            if args.classes and V.CLASSES[g["category_id"]] not in args.classes.split(","):
                continue
            b = best_pred([p for p in p0 if p["category_id"] == g["category_id"]], g["bbox"], 0.5)
            if b is not None or not args.only_m0_ok:
                chosen.append((d, bgr, gi, p0))
                break
        if len(chosen) >= args.n:
            break
    if not chosen:
        raise SystemExit("Aucun exemple retenu (essayez sans --classes ou sans --only-m0-ok).")

    cols = list(models)
    fig, axes = plt.subplots(len(chosen), len(cols), figsize=(3.3 * len(cols), 3.0 * len(chosen)), squeeze=False)
    for r, (d, bgr, gi, p0) in enumerate(chosen):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        g = gt[d["image_id"]][gi]
        window = crop_box(g["bbox"], rgb.shape)
        for c, k in enumerate(cols):
            preds = p0 if k == "M0" else run_model(*models[k], bgr)
            draw_crop(axes[r][c], rgb, window, g["bbox"], best_pred(preds, g["bbox"]),
                      f"{k}" + (f"  (GT: {V.CLASSES[g['category_id']]})" if c == 0 else ""), V.CLASSES)
        print(f"  exemple {r + 1}/{len(chosen)}", flush=True)
    fig.suptitle(f"Pruned then recovered detector - {CONFIGS[args.config][1]} : "
                 f"prediction (red) vs ground truth (green)", y=1.0, fontsize=10)
    fig.tight_layout()
    save(fig, f"examples_{args.config}.png")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("figures")
    sub.add_parser("tables")
    e = sub.add_parser("examples")
    e.add_argument("--config", default="expose_whole_backbone", choices=list(CONFIGS))
    e.add_argument("--n", type=int, default=4)
    e.add_argument("--seed", type=int, default=3)
    e.add_argument("--threads", type=int, default=4)
    e.add_argument("--skip", default="", help="ex. M1 pour ne pas afficher M1")
    e.add_argument("--classes", default="", help="ex. Broken,Flashover damage (par defaut : toutes)")
    e.add_argument("--only-m0-ok", dest="only_m0_ok", action="store_true", default=True)
    e.add_argument("--any", dest="only_m0_ok", action="store_false",
                   help="ne pas se limiter aux isolateurs que M0 detecte correctement")
    args = ap.parse_args()
    {"figures": cmd_figures, "tables": cmd_tables, "examples": cmd_examples}[args.cmd](args)


if __name__ == "__main__":
    main()