"""
M1 conforme au rapport, a partir de M0-8000 UNIQUEMENT.

Trois commandes (a lancer dans cet ordre, depuis E:\\Stage2\\TinyML_Framework) :

  python run_m1_report.py score  [--n-images 300]
        Scores Taylor |w * dL/dw| sur M0-8000 (une seule fois, reprise automatique si interrompu).
        Tous les poids du reseau sont rendus entrainables pour le calcul (FREEZE_AT=0) : sinon le stem
        et res2, geles par defaut dans Detectron2, n'auraient AUCUN score et ne seraient jamais elagues.

  python run_m1_report.py plan   [--config NOM|all]
        Aucune chirurgie. Decouvre les unites couplees, planifie, predit la reduction de parametres,
        affiche le pourcentage de canaux retires par bloc. Rapide : a lire AVANT de couper.

  python run_m1_report.py calibrate
        Aucune chirurgie. Teste plusieurs percentiles d'ecretage et retient celui qui reproduit le mieux
        les cartes de canaux du rapport a compression egale. A lancer avant 'prune'.

  python run_m1_report.py prune  --config NOM [--ratio R | --target-reduction P]
        Chirurgie + portes de controle + sauvegarde du checkpoint M1 et d'un rapport JSON.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import os
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ----------------------------------------------------------------- chemins
ROOT = Path(os.environ.get("TINYML_ROOT", r"E:\Stage2\TinyML_Framework"))
FASTER = Path(os.environ.get("TINYML_FASTER", r"E:\Stage2\Faster-rcnn"))
M0_CHECKPOINT = Path(os.environ.get("TINYML_M0", str(FASTER / "output_8000" / "model_final.pth")))
M0_SHA256 = "740FFEB4C02BE3658DE1E3CC486588BE16B83BC1150592DAE4D69C5984B9BB8E"
IDID = Path(os.environ.get("TINYML_IDID", r"E:\Stage2\IDID DATASET"))
JSON_DIR = Path(os.environ.get("TINYML_JSON_DIR", str(IDID)))
TRAIN_JSON = JSON_DIR / "training_data.json"
IMAGES = Path(os.environ.get("TINYML_IMAGES", str(IDID / "Train_IDID_V1.2" / "Train" / "Images")))
OUT = ROOT / "m1_report"
CKPT_DIR = OUT / "checkpoints"
SCORES_FILE = OUT / "taylor_scores_m0_8000.pt"
PARTIAL_FILE = OUT / "taylor_scores_partial.pt"
WEIGHT_SCORES_FILE = OUT / "taylor_weights_m0_8000.pt"      # scores PAR POIDS (mecanisme zeros)
WEIGHT_PARTIAL_FILE = OUT / "taylor_weights_partial.pt"
TRAIN_NAME = "my_dataset_train"
SEED = 0


# ----------------------------------------------------------------- utilitaires
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest().upper()


def build_cfg():
    from detectron2 import model_zoo
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = str(M0_CHECKPOINT)
    cfg.MODEL.DEVICE = os.environ.get("TINYML_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.BACKBONE.FREEZE_AT = 0                       # tous les poids entrainables (scores Taylor complets)
    cfg.INPUT.RANDOM_FLIP = "none"                         # scores deterministes
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATASETS.TRAIN = (TRAIN_NAME,)
    return cfg


def register_train():
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances

    if TRAIN_NAME in DatasetCatalog.list():
        DatasetCatalog.remove(TRAIN_NAME)
    if TRAIN_NAME in MetadataCatalog.list():
        MetadataCatalog.remove(TRAIN_NAME)
    register_coco_instances(TRAIN_NAME, {}, str(TRAIN_JSON), str(IMAGES))
    MetadataCatalog.get(TRAIN_NAME).thing_classes = ["No issues", "Broken", "Flashover damage"]


def load_m0(cfg):
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model

    if not M0_CHECKPOINT.is_file():
        raise FileNotFoundError(M0_CHECKPOINT)
    digest = sha256(M0_CHECKPOINT)
    if digest != M0_SHA256:
        raise RuntimeError(f"Ce n'est PAS M0-8000 (SHA256 {digest}). Arret.")
    model = build_model(cfg)
    DetectionCheckpointer(model).load(str(M0_CHECKPOINT))
    n = sum(p.numel() for p in model.parameters())
    print(f"[M0] M0-8000 charge, SHA256 OK, parametres = {n:,}", flush=True)
    if n != 41_305_311:
        raise RuntimeError(f"Parametres inattendus : {n} (attendu 41,305,311)")
    return model


def example_inputs():
    return [{"image": torch.zeros((3, 800, 800), dtype=torch.float32), "height": 800, "width": 800}]


def real_image_input():
    import cv2
    from detectron2.data import DatasetCatalog

    d = DatasetCatalog.get(TRAIN_NAME)[0]
    img = cv2.imread(d["file_name"])
    if img is None:
        raise RuntimeError(f"Image illisible : {d['file_name']}")
    h, w = img.shape[:2]
    return [{"image": torch.as_tensor(img.astype("float32").transpose(2, 0, 1)), "height": h, "width": w}]


# ----------------------------------------------------------------- commande : score
def cmd_score(args):
    from detectron2.data import DatasetCatalog, DatasetMapper
    from detectron2.utils.events import EventStorage
    from tinyml.importance.taylor import TaylorImportance

    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = build_cfg()
    register_train()
    model = load_m0(cfg)
    for p in model.parameters():
        p.requires_grad_(True)

    dicts = [d for d in DatasetCatalog.get(TRAIN_NAME) if len(d.get("annotations", [])) > 0]
    random.Random(SEED).shuffle(dicts)
    dicts = dicts[: args.n_images]
    print(f"[SCORE] {len(dicts)} images (graine {SEED}), seuil : sommes |w*grad| par canal", flush=True)

    mapper = DatasetMapper(cfg, is_train=True)
    ti = TaylorImportance(model, loss_fn=lambda m, b: sum(m(b).values()))

    done = 0
    if PARTIAL_FILE.is_file():
        part = torch.load(PARTIAL_FILE, map_location="cpu")
        if part.get("m0_sha") == M0_SHA256 and part.get("seed") == SEED:
            ti.scores = {k: v.clone() for k, v in part["scores"].items()}
            done = int(part["done"])
            print(f"[SCORE] reprise a l'image {done}", flush=True)

    model.train()
    t0 = time.time()
    # Detectron2 exige un EventStorage pour calculer les pertes en mode entrainement
    with EventStorage():
        for i in range(done, len(dicts)):
            batch = [mapper(copy.deepcopy(dicts[i]))]
            ti.accumulate_batch(batch)
            if (i + 1) % 10 == 0 or i + 1 == len(dicts):
                el = time.time() - t0
                print(f"  {i + 1}/{len(dicts)}  ({el / (i + 1 - done):.1f} s/image)", flush=True)
            if (i + 1) % 20 == 0:
                torch.save({"scores": ti.scores, "done": i + 1, "seed": SEED, "m0_sha": M0_SHA256}, PARTIAL_FILE)

    missing = [n for n, m in model.named_modules()
               if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)) and n not in ti.scores]
    if missing:
        print("[SCORE] ATTENTION modules sans score :", missing, flush=True)
    torch.save({
        "scores": ti.scores, "n_images": len(dicts), "seed": SEED, "m0_sha": M0_SHA256,
        "images": [Path(d["file_name"]).name for d in dicts],
    }, SCORES_FILE)
    print(f"[SCORE] ecrit : {SCORES_FILE}  ({len(ti.scores)} modules)", flush=True)


# ----------------------------------------------------------------- commande : score-weights
def cmd_score_weights(args):
    """Scores Taylor PAR POIDS |w * dL/dw| (meme 300 images, meme graine que 'score').
    Necessaire au mecanisme 'poids a zero puis filtres retires' (tables du rapport)."""
    import torch.nn as nn
    from detectron2.data import DatasetCatalog, DatasetMapper
    from detectron2.utils.events import EventStorage

    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = build_cfg()
    register_train()
    model = load_m0(cfg)
    for p_ in model.parameters():
        p_.requires_grad_(True)

    dicts = [d for d in DatasetCatalog.get(TRAIN_NAME) if len(d.get("annotations", [])) > 0]
    random.Random(SEED).shuffle(dicts)
    dicts = dicts[: args.n_images]
    print(f"[SCORE-W] {len(dicts)} images (graine {SEED}), scores |w*grad| PAR POIDS", flush=True)

    mapper = DatasetMapper(cfg, is_train=True)
    mods = {n: m for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))}
    acc = {n: torch.zeros_like(m.weight, dtype=torch.float32) for n, m in mods.items()}
    done = 0
    if WEIGHT_PARTIAL_FILE.is_file():
        part = torch.load(WEIGHT_PARTIAL_FILE, map_location="cpu")
        if part.get("m0_sha") == M0_SHA256 and part.get("seed") == SEED:
            acc = part["scores"]
            done = int(part["done"])
            print(f"[SCORE-W] reprise a l'image {done}", flush=True)

    model.train()
    t0 = time.time()
    with EventStorage():
        for i in range(done, len(dicts)):
            batch = [mapper(copy.deepcopy(dicts[i]))]
            model.zero_grad(set_to_none=True)
            loss = sum(model(batch).values())
            loss.backward()
            for n, m in mods.items():
                if m.weight.grad is None:
                    continue
                c = (m.weight.detach() * m.weight.grad.detach()).abs().float()
                if not torch.isfinite(c).all():
                    raise RuntimeError(f"score non fini : {n}")
                acc[n] += c
            model.zero_grad(set_to_none=True)
            if (i + 1) % 10 == 0 or i + 1 == len(dicts):
                el = time.time() - t0
                print(f"  {i + 1}/{len(dicts)}  ({el / (i + 1 - done):.1f} s/image)", flush=True)
            if (i + 1) % 25 == 0:
                torch.save({"scores": acc, "done": i + 1, "seed": SEED, "m0_sha": M0_SHA256}, WEIGHT_PARTIAL_FILE)

    torch.save({"scores": acc, "n_images": len(dicts), "seed": SEED, "m0_sha": M0_SHA256,
                "images": [Path(d["file_name"]).name for d in dicts]}, WEIGHT_SCORES_FILE)
    print(f"[SCORE-W] ecrit : {WEIGHT_SCORES_FILE}  ({len(acc)} modules)", flush=True)


def load_weight_scores():
    if not WEIGHT_SCORES_FILE.is_file():
        raise FileNotFoundError(f"Lance d'abord :  python run_m1_report.py score-weights\n({WEIGHT_SCORES_FILE})")
    blob = torch.load(WEIGHT_SCORES_FILE, map_location="cpu")
    if blob["m0_sha"] != M0_SHA256:
        raise RuntimeError("Scores par poids calcules sur un autre modele que M0-8000. Refaire 'score-weights'.")
    return blob


# ----------------------------------------------------------------- pruner commun
def load_scores():
    if not SCORES_FILE.is_file():
        raise FileNotFoundError(f"Lance d'abord :  python run_m1_report.py score\n({SCORES_FILE})")
    blob = torch.load(SCORES_FILE, map_location="cpu")
    if blob["m0_sha"] != M0_SHA256:
        raise RuntimeError("Scores calcules sur un autre modele que M0-8000. Refaire 'score'.")
    return blob


def make_pruner(model, scores):
    from tinyml.pruning.report_pruner import ReportPruner

    pr = ReportPruner(model, example_inputs(), scores)
    pr.discover_units()
    return pr


def print_units(pr):
    print("\nUNITES DECOUVERTES (canaux couples) :")
    print(f"{'uid':>3} {'type':9} {'stage':5} {'canaux':>6}  membres (premiers)  | proteges ?")
    for u in pr.units:
        print(f"{u.uid:>3} {u.label:9} {str(u.stage or ''):5} {u.n_channels:>6}  "
              f"{', '.join(u.members[:2])}{' ...' if len(u.members) > 2 else ''}  | {u.protected_reason or ''}")
    if pr.rejected:
        print("\nGROUPES REJETES A LA DECOUVERTE :", json.dumps(pr.rejected, indent=1))


# ----------------------------------------------------------------- commande : plan
def cmd_plan(args):
    from tinyml.pruning.report_policy import CONFIGS, RECOMMENDED_ORDER

    torch.set_num_threads(args.threads)
    cfg = build_cfg()
    register_train()
    model = load_m0(cfg)
    blob = load_scores()
    pr = make_pruner(model, blob["scores"])
    print_units(pr)

    names = RECOMMENDED_ORDER if args.config == "all" else [args.config]
    print("\nPLAN (aucune chirurgie)  :  r litteral  /  r qui donne la meme reduction que le rapport")
    for name in names:
        pol = copy.deepcopy(CONFIGS[name])
        c = parse_clip(args.clip)
        if c != "policy":
            pol.clip = c
        plan, st = pr.plan(pol)
        pred, _ = pr.predict_parameters(plan)
        red = 100.0 * (pr.params_before - pred) / pr.params_before
        line = f"\n== {name}: r={pol.ratio}  ->  reduction predite {red:.1f} %  (rapport : {pol.ref_params_reduced_pct} %)"
        print(line)
        for k, v in pr.summary_by_label(plan).items():
            print(f"     {k:14} -{v['removed_pct']:5.1f} %  ({v['removed']}/{v['channels']})")
        print(f"     [rapport] {pol.ref_text}")
        if pol.ref_params_reduced_pct is not None:
            r_eq = pr.solve_ratio(pol, pol.ref_params_reduced_pct)
            print(f"     r equivalent pour {pol.ref_params_reduced_pct} % : {r_eq:.4f}")


# ----------------------------------------------------------------- utilitaires clip / journal
def parse_clip(txt):
    """'none' -> None ; '5,95' -> (5.0, 95.0) ; absent -> 'policy' (valeur de la configuration)."""
    if txt is None:
        return "policy"
    if txt.lower() == "none":
        return None
    if txt.lower() == "rank":
        return "rank"
    lo, hi = txt.split(",")
    return (float(lo), float(hi))


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for st in self.streams:
            st.write(data)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def enable_log(name):
    """Tout ce qui s'affiche est aussi ecrit dans m1_report\\last_<commande>.log (a joindre au chat)."""
    OUT.mkdir(parents=True, exist_ok=True)
    f = open(OUT / f"last_{name}.log", "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, f)


def deviation(pcts, ranges):
    """Ecart moyen (en points de %) entre les % de canaux retires par unite et les fourchettes du rapport."""
    parts = {}
    for key, (lo, hi) in ranges.items():
        vals = pcts.get(key, [])
        if not vals:
            continue
        d = [0.0 if lo <= v <= hi else (lo - v if v < lo else v - hi) for v in vals]
        parts[key] = (sum(d) / len(d), sum(vals) / len(vals), min(vals), max(vals))
    total = sum(v[0] for v in parts.values()) / max(1, len(parts))
    return total, parts


# ----------------------------------------------------------------- commande : calibrate
def cmd_calibrate(args):
    """Cherche les percentiles d'ecretage qui reproduisent le mieux les cartes de canaux du rapport,
    A COMPRESSION EGALE (ratio ajuste pour retrouver la reduction de parametres du rapport)."""
    from tinyml.pruning.report_policy import CONFIGS

    torch.set_num_threads(args.threads)
    cfg = build_cfg()
    register_train()
    model = load_m0(cfg)
    blob = load_scores()
    pr = make_pruner(model, blob["scores"])

    grid = ["rank", None, (1.0, 99.0), (2.0, 98.0), (5.0, 95.0), (10.0, 90.0), (15.0, 85.0), (25.0, 75.0)]
    names = [n for n, c in CONFIGS.items()
             if c.ref_ranges and c.ref_params_reduced_pct and n != "uniform_local"]
    results = []
    for clip in grid:
        print(f"\n######## ecretage = {clip} ########", flush=True)
        devs = []
        for name in names:
            pol = copy.deepcopy(CONFIGS[name])
            pol.clip = clip
            r = pr.solve_ratio(pol, pol.ref_params_reduced_pct)
            plan, _ = pr.plan(pol, r)
            pred, _ = pr.predict_parameters(plan)
            red = 100.0 * (pr.params_before - pred) / pr.params_before
            dev, parts = deviation(pr.unit_pcts(plan), pol.ref_ranges)
            devs.append(dev)
            detail = "  ".join(
                f"{k}: moy {v[1]:.0f} % [{v[2]:.0f}-{v[3]:.0f}] (rapport {pol.ref_ranges[k][0]}-{pol.ref_ranges[k][1]})"
                for k, v in parts.items())
            print(f"  {name:22} r={r:.3f} red={red:5.1f} % (rapport {pol.ref_params_reduced_pct}) "
                  f"ecart={dev:5.1f} pts | {detail}", flush=True)
        results.append((sum(devs) / len(devs), clip))
        print(f"  ==> ecart moyen = {results[-1][0]:.2f} points", flush=True)
    results.sort(key=lambda t: t[0])
    print("\nCLASSEMENT (meilleur en premier) :")
    for d, c in results:
        print(f"   ecretage {c}  ->  ecart moyen {d:.2f} points")


# ----------------------------------------------------------------- commande : zcalibrate
def cmd_zcalibrate(args):
    """TEST DECISIF du mecanisme 'poids a zero puis filtres retires' : on applique les ratios et seuils
    BRUTS du rapport (aucune cible de reduction imposee) et on compare, pour les 7 configurations,
    la reduction de parametres obtenue a celle du rapport, ainsi que les cartes de canaux par bloc."""
    from tinyml.importance.report_zeros import (GROUP_MODES, channel_scores_from_weights,
                                                normalize_groups, plan_zero_removals)
    from tinyml.pruning.report_policy import CONFIGS

    torch.set_num_threads(args.threads)
    cfg = build_cfg()
    register_train()
    model = load_m0(cfg)
    wblob = load_weight_scores()
    ws = wblob["scores"]
    pr = make_pruner(model, channel_scores_from_weights(ws))
    clips = [(1.0, 99.0), (5.0, 95.0), (10.0, 90.0), None]
    groups = args.groups.split(",") if args.groups else list(GROUP_MODES)
    results = []
    for group in groups:
        for clip in clips:
            print(f"\n######## groupe={group}  ecretage={clip} ########", flush=True)
            norm = normalize_groups(pr.units, ws, group, clip)
            errs, devs = [], []
            for name, base in CONFIGS.items():
                pol = copy.deepcopy(base)
                plan, st, _ = plan_zero_removals(pr.units, norm, pol.weights, pol.ratio,
                                                 pol.filter_zero_thresh, pol.min_keep)
                pred, _ = pr.predict_parameters(plan)
                red = 100.0 * (pr.params_before - pred) / pr.params_before
                err = abs(red - pol.ref_params_reduced_pct)
                dev, parts = deviation(pr.unit_pcts(plan), pol.ref_ranges) if pol.ref_ranges else (0.0, {})
                if name != "uniform_local":
                    errs.append(err)
                    devs.append(dev)
                detail = "  ".join(f"{k}: moy {v[1]:.0f} % [{v[2]:.0f}-{v[3]:.0f}] (rapport {pol.ref_ranges[k][0]}-{pol.ref_ranges[k][1]})"
                                   for k, v in parts.items())
                print(f"  {name:22} r={pol.ratio} s={pol.filter_zero_thresh} reduction={red:5.1f} % "
                      f"(rapport {pol.ref_params_reduced_pct}) err={err:4.1f} pts ecart_blocs={dev:5.1f} | {detail}",
                      flush=True)
            results.append((sum(errs) / len(errs), sum(devs) / len(devs), group, clip))
            print(f"  ==> erreur moyenne de reduction (6 configs) = {results[-1][0]:.2f} pts ; "
                  f"ecart moyen des blocs = {results[-1][1]:.2f} pts", flush=True)
    results.sort(key=lambda t: (t[0], t[1]))
    print("\nCLASSEMENT (erreur de reduction vs rapport, puis ecart des blocs) :")
    for e, d, g, c in results:
        print(f"   groupe={g:10} ecretage={str(c):12} erreur reduction {e:5.2f} pts | ecart blocs {d:5.2f} pts")


# ----------------------------------------------------------------- commande : prune
def cmd_prune(args):
    from tinyml.pruning.report_policy import CONFIGS

    torch.set_num_threads(args.threads)
    if args.config not in CONFIGS:
        raise SystemExit(f"config inconnue ; choix : {list(CONFIGS)}")
    pol = copy.deepcopy(CONFIGS[args.config])
    c = parse_clip(args.clip)
    if c != "policy":
        pol.clip = c
    cfg = build_cfg()
    register_train()
    model = load_m0(cfg)
    mode = getattr(args, "mode", "channels")
    masks = None
    if mode == "zeros":
        from tinyml.importance.report_zeros import (channel_scores_from_weights, normalize_groups,
                                                    plan_zero_removals)
        if args.target_reduction is not None:
            raise SystemExit("--target-reduction n'a pas de sens en mode zeros : on applique le ratio et le seuil du rapport")
        blob = load_weight_scores()
        pr = make_pruner(model, channel_scores_from_weights(blob["scores"]))
    else:
        blob = load_scores()
        pr = make_pruner(model, blob["scores"])
    print_units(pr)

    if mode == "zeros":
        ratio = pol.ratio if args.ratio is None else args.ratio
        zclip = pol.clip
        tag = f"{args.group}_{'none' if zclip is None else '%g-%g' % zclip}"
        norm = normalize_groups(pr.units, blob["scores"], args.group, zclip, pol.min_keep)
        plan, st, masks = plan_zero_removals(pr.units, norm, pol.weights, ratio, pol.filter_zero_thresh, pol.min_keep)
        print(f"[ZEROS] r={ratio} seuil_filtre={pol.filter_zero_thresh} groupe={args.group} ecretage={zclip} : "
              f"{st.get('zeroed', 0):,}/{st.get('pool', 0):,} poids a zero ; seuil de score {st.get('weight_threshold')} ; "
              f"canaux retires={st.get('channels_removed', 0)}", flush=True)
    elif args.ratio is not None:
        ratio, tag = args.ratio, f"r{args.ratio:.3f}"
    elif args.target_reduction is not None:
        ratio = pr.solve_ratio(pol, args.target_reduction)
        tag = f"t{args.target_reduction:.1f}"
        print(f"[RATIO] r={ratio:.4f} pour viser {args.target_reduction} % de parametres en moins", flush=True)
    else:
        ratio, tag = pol.ratio, f"r{pol.ratio:.2f}"

    if mode != "zeros":
        zc_ = pol.clip                                    # le nom du fichier contient l'ecretage (pas d'ecrasement)
        tag += "_c" + ("rank" if zc_ == "rank" else "none" if zc_ is None else "%g-%g" % zc_)
        plan, st = pr.plan(pol, ratio)
    predicted, per_module = pr.predict_parameters(plan)
    print(f"[PLAN] config={pol.name} r={ratio:.4f} canaux retires={st.get('removed', st.get('channels_removed'))}/{st.get('pool')} "
          f"parametres predits={predicted:,}", flush=True)
    before = pr.params_before

    # ------------------------------------------------ chirurgie
    if masks:                       # mode zeros : les poids mis a zero le restent apres la chirurgie
        mods_now = dict(model.named_modules())
        with torch.no_grad():
            for n_, mk in masks.items():
                mods_now[n_].weight[mk] = 0.0
    rep = pr.apply(plan)
    after = pr.count_parameters()
    conv_lin_all = [m_ for m_ in model.modules() if isinstance(m_, (torch.nn.Conv2d, torch.nn.Linear))]
    residual_sparsity = (sum(int((m_.weight == 0).sum()) for m_ in conv_lin_all)
                         / sum(m_.weight.numel() for m_ in conv_lin_all))

    # ------------------------------------------------ portes de controle
    gates = {}
    gates["no_rejected_group"] = not rep["rejected"]
    gates["params_equal_prediction"] = (after == predicted)
    gates["fewer_params"] = after < before
    model.eval()
    try:
        with torch.no_grad():
            out = model(real_image_input())
        inst = out[0]["instances"]
        finite = all(torch.isfinite(v).all().item() for v in inst.get_fields().values() if torch.is_tensor(v))
        gates["forward_finite"] = bool(finite)
        gates["n_detections_at_0.05_on_one_image"] = int(len(inst))
    except Exception as exc:                                  # noqa: BLE001
        gates["forward_finite"] = False
        gates["forward_error"] = repr(exc)
    gates["finite_parameters"] = all(torch.isfinite(p).all().item() for p in model.parameters())

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = "M1Z" if mode == "zeros" else "M1R"
    path = CKPT_DIR / f"{prefix}_{pol.name}_{tag}.pth"
    report = {
        "mode": mode, "group": getattr(args, "group", None), "residual_sparsity_after_surgery": residual_sparsity,
        "config": pol.name, "ratio_used": ratio, "weights": pol.weights, "min_keep": pol.min_keep,
        "filter_zero_thresh": pol.filter_zero_thresh, "percentile_clip": pol.clip,
        "m0_checkpoint": str(M0_CHECKPOINT), "m0_sha256": M0_SHA256,
        "taylor_n_images": blob["n_images"],
        "params_before": before, "params_after": after, "params_predicted": predicted,
        "reduction_pct": 100.0 * (before - after) / before,
        "report_reference": {"params_reduced_pct": pol.ref_params_reduced_pct, "ref_text": pol.ref_text,
                             "ap_after_recovery": pol.ref_ap, "ap50_after_recovery": pol.ref_ap50},
        "summary_by_label": pr.summary_by_label(plan),
        "channel_map": pr.channel_map(plan),
        "rejected": rep["rejected"], "gates": gates,
    }
    torch.save({"model_state_dict": model.state_dict(), "pruning_report": report,
                "source": "M0-8000", "created": time.strftime("%Y-%m-%d %H:%M:%S")}, path)

    # taille du fichier = 4 octets x parametres + buffers (si identique a M0 : pas d'elagage reel)
    size = path.stat().st_size
    gates["checkpoint_bytes"] = size
    gates["checkpoint_size_matches_params"] = abs(size - (4 * after)) < 0.02 * 4 * after + 2_000_000

    # test de rechargement avec la meme reconstruction que l'evaluation
    try:
        from evaluate_m0_m1_pruned import reconstruct_pruned_architecture
        from detectron2.modeling import build_model
        fresh = build_model(build_cfg())
        sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
        reconstruct_pruned_architecture(fresh, sd)
        fresh.load_state_dict(sd, strict=True)
        gates["reload_ok"] = True
    except Exception as exc:                                  # noqa: BLE001
        gates["reload_ok"] = False
        gates["reload_error"] = repr(exc)[:500]

    report["gates"] = gates
    (path.with_suffix(".report.json")).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 70)
    if mode == "zeros":
        print(f"[mode zeros] sparsite residuelle des poids survivants : {100 * residual_sparsity:.1f} %")
    print(f"M1 [{pol.name}]  parametres {before:,} -> {after:,}   (-{report['reduction_pct']:.2f} %)")
    print(f"rapport (apres recuperation) : -{pol.ref_params_reduced_pct} %   {pol.ref_text}")
    for k, v in pr.summary_by_label(plan).items():
        print(f"   {k:14} -{v['removed_pct']:5.1f} %")
    print("PORTES :", json.dumps({k: v for k, v in gates.items()}, indent=1))
    ok = all(gates[k] for k in ("no_rejected_group", "params_equal_prediction", "fewer_params",
                                "forward_finite", "finite_parameters", "reload_ok"))
    print("RESULTAT :", "M1 VALIDE" if ok else "M1 NON VALIDE - ne pas utiliser, envoyer ce journal")
    print(f"checkpoint : {path}")
    print("=" * 70)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("--n-images", type=int, default=300)
    s.add_argument("--threads", type=int, default=4); s.set_defaults(fn=cmd_score)
    p = sub.add_parser("plan"); p.add_argument("--config", default="all")
    p.add_argument("--clip", default=None, help="ex: 5,95 ou none")
    p.add_argument("--threads", type=int, default=4); p.set_defaults(fn=cmd_plan)
    c = sub.add_parser("calibrate"); c.add_argument("--threads", type=int, default=4)
    c.set_defaults(fn=cmd_calibrate)
    sw = sub.add_parser("score-weights"); sw.add_argument("--n-images", type=int, default=300)
    sw.add_argument("--threads", type=int, default=4); sw.set_defaults(fn=cmd_score_weights)
    zc = sub.add_parser("zcalibrate"); zc.add_argument("--threads", type=int, default=4)
    zc.add_argument("--groups", default=None, help="ex: layer,unit (defaut : layer,unit,label,labelstage)")
    zc.set_defaults(fn=cmd_zcalibrate)
    r = sub.add_parser("prune"); r.add_argument("--config", required=True)
    g = r.add_mutually_exclusive_group()
    g.add_argument("--ratio", type=float); g.add_argument("--target-reduction", type=float)
    r.add_argument("--clip", default=None, help="ex: 5,95 ou none")
    r.add_argument("--mode", choices=["channels", "zeros"], default="channels",
                   help="channels : classement par canaux (texte 4.4) ; zeros : poids a zero puis filtres retires (tables du rapport)")
    r.add_argument("--group", default="layer", choices=["layer", "unit", "label", "labelstage"],
                   help="mode zeros : groupe de normalisation")
    r.add_argument("--threads", type=int, default=4); r.set_defaults(fn=cmd_prune)
    args = ap.parse_args()
    enable_log(args.cmd + (f"_{args.config}" if getattr(args, "config", None) else "") + ("_zeros" if getattr(args, "mode", "") == "zeros" else "")
               + (f"_clip{args.clip.replace(',', '-')}" if getattr(args, "clip", None) else ""))
    args.fn(args)


if __name__ == "__main__":
    main()