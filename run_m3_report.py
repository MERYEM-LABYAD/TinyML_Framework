"""
M3 conforme au rapport : DISTILLATION de features (FPN P2-P5) pour recuperer la precision perdue par le pruning.

Rapport, 4.6 : "The teacher is the original model, frozen; gradients flow only into the pruned student. [...]
Feature distillation matches internal feature maps with a mean-squared-error term on a shape-stable bus:
        L = task_loss + beta * MSE(student_features, teacher_features)
It is the natural choice for the detector, where the fixed-width FPN pyramid (levels P2-P5) provides matching
shapes on teacher and student without an adapter."   Et 5.1 : beta = 0.5.

CE QUE LE RAPPORT FIXE      : teacher = M0-8000 gele ; features = FPN P2, P3, P4, P5 ; beta = 0.5 ; perte = perte de
                              detection + beta * MSE.
CE QUE LE RAPPORT NE FIXE PAS (choix a DECLARER, modifiables en ligne de commande) :
                              optimiseur (SGD, momentum 0.9, weight decay 1e-4), learning rate (1e-3, warmup 50 iter
                              puis cosinus), taille de batch (2), nombre d'iterations (800), augmentation (flip
                              horizontal, multi-echelle 480-576 px au lieu de 800 px pour tenir sur CPU), stem et res2
                              geles (defaut Detectron2), clipping de gradient (norme 10), images (jeu d'entrainement).

L'etudiant part de M2 (M1 + FrozenBN recalibres). Les FrozenBN restent gelees (ce sont des buffers, pas des
parametres) ; tous les poids de convolution et de lineaire sont entrainables (FREEZE_AT = 0).

Commandes (depuis E:\\Stage2\\TinyML_Framework) :
  python run_m3_report.py distill --m2 m1_report\\checkpoints\\M2_expose_whole_backbone_t41.8_crank_matched.pth
        (reprise automatique si interrompu : relancer la MEME commande)
  python run_m3_report.py curves  --m2 m1_report\\checkpoints\\M2_expose_whole_backbone_t41.8_crank_matched.pth
        (courbes de recuperation PNG + CSV, equivalent des figures 20, 22, ... du rapport)
  python evaluate_m1_report.py --checkpoint m1_report\\checkpoints\\M3_expose_whole_backbone_t41.8_crank_matched.pth
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_m1_report as R                                   # noqa: E402

LEVELS = ("p2", "p3", "p4", "p5")


def m3_paths(m2_path: Path, tag: str = ""):
    stem = m2_path.stem
    base = stem[3:] if stem.startswith("M2_") else stem
    if tag:
        base = f"{base}_{tag}"
    out = R.CKPT_DIR / f"M3_{base}.pth"
    wip = R.CKPT_DIR / f"M3_{base}.wip.pth"
    csv_path = R.OUT / f"m3_losses_{base}.csv"
    return out, wip, csv_path


def lr_at(it, total, base_lr, warmup):
    if it < warmup:
        return base_lr * (0.001 + 0.999 * it / max(1, warmup))
    p = (it - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * p))


def load_student(cfg, m2_path):
    from detectron2.modeling import build_model
    from evaluate_m0_m1_pruned import reconstruct_pruned_architecture

    blob = torch.load(m2_path, map_location="cpu", weights_only=False)
    sd = blob["model_state_dict"]
    report = blob["pruning_report"]
    if report.get("m0_sha256") != R.M0_SHA256:
        raise SystemExit("Ce M2 ne vient pas de M0-8000 (SHA256 different). Arret.")
    if blob.get("stage") != "M2":
        raise SystemExit("Ce fichier n'est pas un M2 (recalibration). Lancez d'abord run_m2_report.py.")
    student = build_model(cfg)
    reconstruct_pruned_architecture(student, sd)
    student.load_state_dict(sd, strict=True)
    return student, report


def training_step(student, teacher, batch, beta):
    """Une iteration : perte de detection du student + beta * MSE sur FPN P2-P5 (teacher gele)."""
    images = student.preprocess_image(batch)
    gt_instances = [x["instances"].to(student.device) for x in batch]
    features = student.backbone(images.tensor)
    with torch.no_grad():
        t_features = teacher.backbone(images.tensor)
    proposals, proposal_losses = student.proposal_generator(images, features, gt_instances)
    _, detector_losses = student.roi_heads(images, features, proposals, gt_instances)
    det = {}
    det.update(detector_losses)
    det.update(proposal_losses)
    task = sum(det.values())
    feat = sum(torch.nn.functional.mse_loss(features[k], t_features[k]) for k in LEVELS) / len(LEVELS)
    total = task + beta * feat
    return total, task, feat, det


def training_step_kd(student, teacher, batch, beta, kd):
    """Comme training_step + (optionnel) KL sur les logits de classification des RoI (teacher gele),
    calcules sur les MEMES boites echantillonnees par le student (hooks sur box_pooler / box_predictor)."""
    import torch.nn.functional as F
    kd["cap"].clear()
    images = student.preprocess_image(batch)
    gt_instances = [x["instances"].to(student.device) for x in batch]
    features = student.backbone(images.tensor)
    with torch.no_grad():
        t_features = teacher.backbone(images.tensor)
    proposals, proposal_losses = student.proposal_generator(images, features, gt_instances)
    _, detector_losses = student.roi_heads(images, features, proposals, gt_instances)
    det = {}
    det.update(detector_losses)
    det.update(proposal_losses)
    task = sum(det.values())
    feat = sum(F.mse_loss(features[k], t_features[k]) for k in LEVELS) / len(LEVELS)
    total = task + beta * feat
    if kd["w"] > 0:
        boxes, s_scores = kd["cap"]["boxes"], kd["cap"]["scores"]
        with torch.no_grad():
            rh = teacher.roi_heads
            tf = rh.box_pooler([t_features[f] for f in rh.box_in_features], boxes)
            t_scores, _ = rh.box_predictor(rh.box_head(tf))
        T = kd["T"]
        kd_loss = F.kl_div(F.log_softmax(s_scores / T, dim=-1), F.softmax(t_scores / T, dim=-1),
                           reduction="batchmean") * T * T
        total = total + kd["w"] * kd_loss
        det["kd_cls_loss"] = kd_loss.detach()
    return total, task, feat, det


def cmd_distill(args):
    from detectron2.data import DatasetCatalog, DatasetMapper
    from detectron2.utils.events import EventStorage

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    m2_path = Path(args.m2)
    out_path, wip_path, csv_path = m3_paths(m2_path, args.tag)
    R.CKPT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = R.build_cfg()
    cfg.INPUT.RANDOM_FLIP = "horizontal"                 # augmentation d'entrainement (non precisee par le rapport)
    sizes = tuple(int(x) for x in args.min_size.split(","))
    cfg.INPUT.MIN_SIZE_TRAIN = sizes                     # multi-echelle reduite : l'entrainement CPU est trop lent a 800 px
    cfg.INPUT.MAX_SIZE_TRAIN = args.max_size
    R.register_train()
    teacher = R.load_m0(cfg)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    student, report = load_student(cfg, m2_path)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"[M3] student = {m2_path.name} : {n_params:,} parametres ({report['config']}, "
          f"-{report['reduction_pct']:.2f} %)", flush=True)
    frozen_prefix = {0: (), 1: ("backbone.bottom_up.stem",),
                     2: ("backbone.bottom_up.stem", "backbone.bottom_up.res2")}[args.freeze_at]
    for name, p in student.named_parameters():
        p.requires_grad_(not name.startswith(frozen_prefix) if frozen_prefix else True)
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[M3] parametres entrainables : {n_train:,} / {n_params:,}  (freeze-at={args.freeze_at} : "
          f"{'tout est entrainable' if not frozen_prefix else 'stem et/ou res2 geles'}) ; tailles d'entree {sizes} "
          f"(max {args.max_size})", flush=True)
    student.train()
    params = [p for p in student.parameters() if p.requires_grad]
    kd = {"w": args.kd_cls, "T": args.kd_temp, "cap": {}}
    hooks = []
    if args.kd_cls > 0:
        hooks.append(student.roi_heads.box_pooler.register_forward_pre_hook(lambda m, a: kd["cap"].__setitem__("boxes", a[1])))
        hooks.append(student.roi_heads.box_predictor.register_forward_hook(lambda m, a, o: kd["cap"].__setitem__("scores", o[0])))
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)

    dicts = [d for d in DatasetCatalog.get(R.TRAIN_NAME) if len(d.get("annotations", [])) > 0]
    mapper = DatasetMapper(cfg, is_train=True)
    n = len(dicts)
    print(f"[M3] {n} images d'entrainement ; batch={args.batch} ; iterations={args.iters} ; lr={args.lr} ; "
          f"beta={args.beta} ; wd={args.wd}", flush=True)

    start = 0
    rows = []
    if wip_path.is_file():
        w = torch.load(wip_path, map_location="cpu", weights_only=False)
        if w.get("args_sig") == (args.iters, args.batch, args.lr, args.beta, args.wd, args.seed):
            student.load_state_dict(w["model_state_dict"], strict=True)
            opt.load_state_dict(w["optimizer"])
            start = int(w["iter"])
            rows = w["rows"]
            print(f"[M3] reprise a l'iteration {start}", flush=True)
        else:
            print("[M3] fichier de reprise ignore (parametres differents)", flush=True)

    def sample_index(k):
        ep, pos = divmod(k, n)
        order = list(range(n))
        random.Random(args.seed * 1000 + ep).shuffle(order)
        return order[pos]

    t0 = time.time()
    with EventStorage():
        for it in range(start, args.iters):
            lr = lr_at(it, args.iters, args.lr, args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            batch = [mapper(copy.deepcopy(dicts[sample_index(it * args.batch + j)])) for j in range(args.batch)]
            total, task, feat, det = training_step_kd(student, teacher, batch, args.beta, kd)
            if not torch.isfinite(total):
                raise SystemExit(f"Perte non finie a l'iteration {it} : arret (diminuer --lr).")
            opt.zero_grad(set_to_none=True)
            total.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
            opt.step()
            rows.append({"iter": it + 1, "lr": lr, "total": float(total), "task": float(task), "feat_mse": float(feat),
                         "grad_norm": gn, **{k: float(v) for k, v in det.items()}})
            done = it + 1 - start
            if (it + 1) % args.log_every == 0 or it + 1 == args.iters:
                el = time.time() - t0
                eta = el / done * (args.iters - it - 1)
                print(f"  it {it + 1}/{args.iters}  loss {float(total):.4f}  task {float(task):.4f}  "
                      f"feat {float(feat):.4f}  lr {lr:.2e}  gn {gn:.1f}  ({el / done:.1f} s/it, reste ~{eta / 3600:.1f} h)",
                      flush=True)
            if (it + 1) % args.save_every == 0 and it + 1 < args.iters:
                torch.save({"model_state_dict": student.state_dict(), "optimizer": opt.state_dict(), "iter": it + 1,
                            "rows": rows, "args_sig": (args.iters, args.batch, args.lr, args.beta, args.wd, args.seed)},
                           wip_path)
                _write_csv(csv_path, rows)

    _write_csv(csv_path, rows)
    # ------------------------------------------------ portes de controle
    for h_ in hooks:
        h_.remove()
    gates = {}
    student.eval()
    gates["params_unchanged"] = (sum(p.numel() for p in student.parameters()) == report["params_after"])
    gates["finite_parameters"] = all(torch.isfinite(p).all().item() for p in student.parameters())
    kk = max(1, min(20, len(rows) // 4))
    first = sum(r["total"] for r in rows[:kk]) / kk
    last = sum(r["total"] for r in rows[-kk:]) / kk
    gates["loss_decreased"] = last < first
    try:
        with torch.no_grad():
            inst = student(R.real_image_input())[0]["instances"]
        gates["forward_finite"] = all(torch.isfinite(v).all().item() for v in inst.get_fields().values()
                                      if torch.is_tensor(v))
        gates["n_detections_at_0.05_on_one_image"] = int(len(inst))
    except Exception as exc:                                  # noqa: BLE001
        gates["forward_finite"] = False
        gates["forward_error"] = repr(exc)[:300]
    new_report = dict(report)
    new_report["distillation"] = {
        "levels": list(LEVELS), "beta": args.beta, "iters": args.iters, "batch": args.batch, "lr": args.lr,
        "warmup": args.warmup, "weight_decay": args.wd, "momentum": 0.9, "grad_clip": args.clip, "seed": args.seed,
        "schedule": "warmup lineaire puis cosinus", "augmentation": "flip horizontal", "min_size_train": list(sizes), "max_size_train": args.max_size, "freeze_at": args.freeze_at, "teacher": "M0-8000 gele",
        "source_m2": str(m2_path), "kd_cls": args.kd_cls, "kd_temp": args.kd_temp, "tag": args.tag, "loss_first20": first, "loss_last20": last, "seconds": time.time() - t0,
    }
    torch.save({"model_state_dict": student.state_dict(), "pruning_report": new_report, "source": "M0-8000",
                "stage": "M3", "created": time.strftime("%Y-%m-%d %H:%M:%S")}, out_path)
    try:
        from detectron2.modeling import build_model
        from evaluate_m0_m1_pruned import reconstruct_pruned_architecture
        fresh = build_model(R.build_cfg())
        sd2 = torch.load(out_path, map_location="cpu", weights_only=False)["model_state_dict"]
        reconstruct_pruned_architecture(fresh, sd2)
        fresh.load_state_dict(sd2, strict=True)
        gates["reload_ok"] = True
    except Exception as exc:                                  # noqa: BLE001
        gates["reload_ok"] = False
        gates["reload_error"] = repr(exc)[:300]
    out_path.with_suffix(".distill.json").write_text(json.dumps(new_report["distillation"] | {"gates": gates},
                                                                indent=2, default=str), encoding="utf-8")
    if wip_path.is_file():
        wip_path.unlink()
    print("\n" + "=" * 70)
    print(f"M3 [{report['config']}]  parametres = {n_params:,}   perte moyenne : {first:.4f} -> {last:.4f}")
    print("PORTES :", json.dumps(gates, indent=1))
    ok = all(gates[k] for k in ("params_unchanged", "finite_parameters", "loss_decreased", "forward_finite", "reload_ok"))
    print("RESULTAT :", "M3 VALIDE" if ok else "M3 NON VALIDE - envoyer ce journal")
    print(f"checkpoint : {out_path}")
    print("Suite : python evaluate_m1_report.py --checkpoint", out_path)
    print("=" * 70)


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def cmd_curves(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, _, csv_path = m3_paths(Path(args.m2), getattr(args, "tag", ""))
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    it = [int(r["iter"]) for r in rows]

    def smooth(v, k=20):
        out, s = [], 0.0
        for i, x in enumerate(v):
            s += x
            if i >= k:
                s -= v[i - k]
            out.append(s / min(i + 1, k))
        return out

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name, a, title in (("total", ax[0], "perte totale"), ("task", ax[1], "perte de detection"),
                           ("feat_mse", ax[2], "MSE features FPN P2-P5")):
        v = [float(r[name]) for r in rows]
        a.plot(it, v, alpha=0.25)
        a.plot(it, smooth(v), lw=2)
        a.set_title(title)
        a.set_xlabel("iteration")
        a.grid(alpha=0.3)
    fig.tight_layout()
    png = csv_path.with_suffix(".png")
    fig.savefig(png, dpi=130)
    print("courbes :", png)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("distill")
    d.add_argument("--m2", required=True)
    d.add_argument("--iters", type=int, default=800)
    d.add_argument("--min-size", default="480,512,544,576", help="tailles d'entree d'entrainement (multi-echelle)")
    d.add_argument("--max-size", type=int, default=960)
    d.add_argument("--freeze-at", type=int, default=2, choices=[0, 1, 2],
                   help="0 : tout entrainable ; 1 : stem gele ; 2 : stem + res2 geles (defaut Detectron2)")
    d.add_argument("--batch", type=int, default=2)
    d.add_argument("--lr", type=float, default=1e-3)
    d.add_argument("--wd", type=float, default=1e-4)
    d.add_argument("--beta", type=float, default=0.5, help="rapport : 0.5")
    d.add_argument("--warmup", type=int, default=50)
    d.add_argument("--clip", type=float, default=10.0)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--log-every", type=int, default=10)
    d.add_argument("--save-every", type=int, default=50)
    d.add_argument("--threads", type=int, default=4)
    d.add_argument("--tag", default="", help="suffixe du nom de sortie (variantes d'ablation)")
    d.add_argument("--kd-cls", type=float, default=0.0, help="poids de la KL sur les logits de classification (0 = desactive, comme le M3 d'origine)")
    d.add_argument("--kd-temp", type=float, default=2.0)
    d.set_defaults(fn=cmd_distill)
    c = sub.add_parser("curves")
    c.add_argument("--m2", required=True)
    c.add_argument("--tag", default="")
    c.set_defaults(fn=cmd_curves)
    args = ap.parse_args()
    R.enable_log("m3_" + args.cmd)
    args.fn(args)


if __name__ == "__main__":
    main()