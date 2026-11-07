"""
gallery_int8.py : galerie ALEATOIRE (non triee) M0 | M3 float | M4 INT8, avec verdict OK / MISS par isolateur.

A copier dans E:\\Stage2\\TinyML_Framework (a cote de run_m4_report.py et report_visuals.py), puis UNE ligne :

  python gallery_int8.py --m3 m1_report\\checkpoints\\M3_maximal_t61.7_crank_matched_v5pure.pth --per-class 3 --seed 7 --threads 4

Ce que fait le script (CPU uniquement, aucune GPU) :
  1. charge M0-8000 et le M3 (float) ;
  2. re-quantifie le M3 en INT8 avec EXACTEMENT les memes reglages que run_m4_report.py
     (200 images d'entrainement, graine 1, meme moteur, stem laisse en FP32 comme --fp32-units stem) : le state_dict INT8 n'est pas rechargeable,
     on refait donc la quantification dans le script (~15-20 min de calibration) ;
  3. tire AU HASARD (graine fixe) per-class isolateurs par classe dans le jeu de test : on ne choisit PAS les reussites ;
  4. affiche pour chaque isolateur : verite terrain en VERT, meilleure prediction en ROUGE, et OK/MISS
     (OK = une prediction de la bonne classe avec IoU >= 0.5, seuil de score 0.5).
Sortie : m1_report\\figures\\gallery_int8_<graine>.png + gallery_int8_<graine>.json (regle de tirage + compte OK/MISS).
Non teste sur ta machine : si une erreur apparait, envoie-moi la trace.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import run_m1_report as R
import run_m4_report as Q
import verify_m0_8000 as V
import report_visuals as RV
from detectron2.data import DatasetCatalog, DatasetMapper
from detectron2.modeling import build_model
from evaluate_m0_m1_pruned import reconstruct_pruned_architecture


def build_m3(path: Path):
    cfg = V.build_cfg(path)
    model = build_model(cfg)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob["model_state_dict"]
    reconstruct_pruned_architecture(model, sd)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3", required=True)
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--backend", default="auto")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    RV.style()
    m3_path = Path(args.m3)
    backend = Q.pick_backend(args.backend)
    if not hasattr(Q, "SKIP_UNITS"):
        raise SystemExit("run_m4_report.py n'est pas patche : lance d'abord  python m4_patch.py run_m4_report.py")
    Q.SKIP_UNITS.clear()
    Q.SKIP_UNITS.add("stem")          # recette finale : le stem reste en FP32 (le quantifier seul coute 20,6 pts d'AP)

    # modeles float : M0 et M3
    models = RV.load_models({"M0": V.DEFAULT_CHECKPOINT, "M3": m3_path}, V)
    # modele a quantifier (copie independante du M3 float)
    qm, qcfg = build_m3(m3_path)
    R.register_train()
    V.register()
    dicts = [d for d in DatasetCatalog.get(R.TRAIN_NAME) if len(d.get("annotations", [])) > 0]
    random.Random(Q.SEED_CALIB).shuffle(dicts)
    dicts = dicts[:args.n_calib]
    print(f"[galerie] quantification INT8 ({len(dicts)} images de calibration)...", flush=True)
    qmodel, _ = Q.quantize_model(qm, DatasetMapper(qcfg, is_train=False), dicts, backend=backend, check_input=None)
    models["M4_INT8"] = (qmodel, qcfg)

    # tirage aleatoire stratifie par classe (graine fixe)
    gt = V.ground_truth_by_image()
    data = [d for d in DatasetCatalog.get(V.DATASET) if gt.get(d["image_id"])]
    random.Random(args.seed).shuffle(data)
    picks, used = [], set()
    for cid in range(len(V.CLASSES)):
        n = 0
        for d in data:
            if d["image_id"] in used:
                continue
            gi = next((i for i, g in enumerate(gt[d["image_id"]]) if g["category_id"] == cid), None)
            if gi is None:
                continue
            picks.append((d, gi))
            used.add(d["image_id"])
            n += 1
            if n >= args.per_class:
                break

    cols = list(models)
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(3.4 * len(cols), 3.0 * len(picks)), squeeze=False)
    summary = {k: {"OK": 0, "MISS": 0} for k in cols}
    rows = []
    for r, (d, gi) in enumerate(picks):
        bgr = cv2.imread(d["file_name"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        g = gt[d["image_id"]][gi]
        window = RV.crop_box(g["bbox"], rgb.shape)
        row = {"image": Path(d["file_name"]).name, "gt_class": V.CLASSES[g["category_id"]]}
        for c, k in enumerate(cols):
            model, cfg = models[k]
            preds = RV.run_model(model, cfg, bgr)
            ok = RV.best_pred([p for p in preds if p["category_id"] == g["category_id"]], g["bbox"], 0.5) is not None
            summary[k]["OK" if ok else "MISS"] += 1
            row[k] = "OK" if ok else "MISS"
            RV.draw_crop(axes[r][c], rgb, window, g["bbox"], RV.best_pred(preds, g["bbox"]),
                         f"{k}  {'OK' if ok else 'MISS'}" + (f"  (GT: {V.CLASSES[g['category_id']]})" if c == 0 else ""),
                         V.CLASSES)
        rows.append(row)
        print(f"  isolateur {r + 1}/{len(picks)}  {row}", flush=True)
    fig.suptitle(f"Random draw (seed {args.seed}, {args.per_class} per class), score >= 0.5: "
                 f"prediction (red) vs ground truth (green)", y=1.0, fontsize=10)
    fig.tight_layout()
    RV.save(fig, f"gallery_int8_{args.seed}.png")
    (RV.FIG / f"gallery_int8_{args.seed}.json").write_text(
        json.dumps({"rule": "random draw, fixed seed, stratified by class, no manual selection",
                    "seed": args.seed, "per_class": args.per_class, "m3": str(m3_path),
                    "summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print("\nResume OK / MISS par modele :", json.dumps(summary))


if __name__ == "__main__":
    main()