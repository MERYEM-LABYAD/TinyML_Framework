"""
ETAPE 1 - Verification de M0-8000 (aucun entrainement, aucune modification de modele).

Corrige les 3 erreurs de evaluate_m0_m1_pruned.py :
  1. Verite terrain lue dans le JSON COCO (DatasetCatalog), plus dans le loader de test (vide).
  2. COCO AP evalue avec SCORE_THRESH_TEST = 0.05 (protocole de l'entrainement).
     Les metriques TP/FP/FN/Precision/Recall sont calculees a part, a 0.5.
  3. Checkpoint = output_8000/model_final.pth (SHA256 attendu : 740FFEB4...).

Porte de controle : AP ~ 71.19 / AP50 ~ 92.15 / AP75 ~ 85.51.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model

# ----------------------------------------------------------------- chemins
FASTER_RCNN_ROOT = Path(r"E:\Stage2\Faster-rcnn")
TEST_JSON = Path(r"E:\Stage2\IDID DATASET\testing_data.json")
TEST_IMAGES = Path(r"E:\Stage2\IDID DATASET\Train_IDID_V1.2\Train\Images")
DEFAULT_CHECKPOINT = FASTER_RCNN_ROOT / "output_8000" / "model_final.pth"
DEFAULT_OUT = FASTER_RCNN_ROOT / "VERIFY_M0_8000"
EXPECTED_SHA256_8000 = "740FFEB4C02BE3658DE1E3CC486588BE16B83BC1150592DAE4D69C5984B9BB8E"

DATASET = "my_dataset_test"
CLASSES = ["No issues", "Broken", "Flashover damage"]
COCO_SCORE_THRESH = 0.05      # protocole COCO (identique a l'entrainement)
CUSTOM_SCORE_THRESH = 0.5     # seuil operationnel pour TP/FP/FN
IOU_THRESH = 0.5

# valeurs a retrouver (log de l'entrainement 5000 -> 8000)
GATE = {"AP": 71.1882, "AP50": 92.1516, "AP75": 85.5056}
GATE_TOLERANCE = 0.3          # points d'AP


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def register() -> None:
    if DATASET in DatasetCatalog.list():
        DatasetCatalog.remove(DATASET)
    if DATASET in MetadataCatalog.list():
        MetadataCatalog.remove(DATASET)
    register_coco_instances(DATASET, {}, str(TEST_JSON), str(TEST_IMAGES))
    MetadataCatalog.get(DATASET).thing_classes = CLASSES


def build_cfg(checkpoint: Path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = str(checkpoint)
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = COCO_SCORE_THRESH
    cfg.DATASETS.TEST = (DATASET,)
    cfg.DATALOADER.NUM_WORKERS = 0
    return cfg


def ground_truth_by_image():
    """GT reelle, depuis le JSON COCO enregistre (format XYWH)."""
    gt = defaultdict(list)
    for d in DatasetCatalog.get(DATASET):
        for a in d.get("annotations", []):
            box = BoxMode.convert(a["bbox"], a["bbox_mode"], BoxMode.XYWH_ABS)
            gt[d["image_id"]].append({"bbox": box, "category_id": int(a["category_id"])})
    return gt


def iou_xywh(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts):
    preds = sorted(preds, key=lambda p: p["score"], reverse=True)
    used, tp, fp = set(), 0, 0
    for p in preds:
        best, best_i = 0.0, None
        for i, g in enumerate(gts):
            if i in used or g["category_id"] != p["category_id"]:
                continue
            v = iou_xywh(p["bbox"], g["bbox"])
            if v > best:
                best, best_i = v, i
        if best_i is not None and best >= IOU_THRESH:
            tp += 1
            used.add(best_i)
        else:
            fp += 1
    return tp, fp, len(gts) - len(used)


def run(model, cfg, out_dir: Path):
    """UNE seule passe d'inference a 0.05 : sert au COCO AP et aux metriques a 0.5."""
    gt = ground_truth_by_image()
    n_gt = sum(len(v) for v in gt.values())
    print(f"[GT] images={len(gt)} annotations={n_gt}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    evaluator = COCOEvaluator(DATASET, output_dir=str(out_dir))
    loader = build_detection_test_loader(cfg, DATASET, num_workers=0)

    tp = fp = fn = 0
    evaluator.reset()
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        for k, inputs in enumerate(loader, 1):
            outputs = model(inputs)
            evaluator.process(inputs, outputs)
            for inp, out in zip(inputs, outputs):
                inst = out["instances"].to("cpu")
                preds = []
                for i in range(len(inst)):
                    s = float(inst.scores[i])
                    if s < CUSTOM_SCORE_THRESH:
                        continue
                    x1, y1, x2, y2 = inst.pred_boxes.tensor[i].tolist()
                    preds.append({"bbox": [x1, y1, x2 - x1, y2 - y1],
                                  "category_id": int(inst.pred_classes[i]), "score": s})
                a, b, c = match(preds, gt.get(inp["image_id"], []))
                tp, fp, fn = tp + a, fp + b, fn + c
            if k % 20 == 0:
                print(f"  {k}/{len(loader)} images", flush=True)
    elapsed = time.perf_counter() - t0

    coco = evaluator.evaluate()["bbox"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    jac = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "coco_score_thresh": COCO_SCORE_THRESH,
        "custom_score_thresh": CUSTOM_SCORE_THRESH,
        "AP": coco["AP"], "AP50": coco["AP50"], "AP75": coco["AP75"],
        "APs": coco.get("APs"), "APm": coco.get("APm"), "APl": coco.get("APl"),
        "AP_per_class": {k: v for k, v in coco.items() if k.startswith("AP-")},
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": prec, "Recall": rec, "F1": f1, "Jaccard": jac,
        "n_gt": n_gt, "seconds_total": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    ckpt, out = Path(args.checkpoint), Path(args.out)

    for p in (TEST_JSON, TEST_IMAGES, ckpt):
        if not p.exists():
            raise FileNotFoundError(p)

    digest = sha256(ckpt)
    print(f"[CKPT] {ckpt}\n[SHA256] {digest}", flush=True)
    if ckpt == DEFAULT_CHECKPOINT and digest != EXPECTED_SHA256_8000:
        raise RuntimeError("Ce n'est PAS le M0-8000 attendu (SHA256 different). Arret.")

    register()
    cfg = build_cfg(ckpt)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(str(ckpt))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] parametres = {n_params:,} (attendu 41,305,311)", flush=True)

    res = run(model, cfg, out)
    res.update({"checkpoint": str(ckpt), "sha256": digest, "parameters": n_params})
    (out / "verify_m0_8000.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    for k in ("AP", "AP50", "AP75", "TP", "FP", "FN", "Precision", "Recall", "F1", "Jaccard"):
        print(f"{k:10s} {res[k]}")
    print("=" * 60)
    ok = all(abs(res[k] - GATE[k]) <= GATE_TOLERANCE for k in GATE)
    print("PORTE DE CONTROLE :", "OK - M0-8000 valide" if ok else
          "ECART - NE PAS CONTINUER, envoyer ce resultat")


if __name__ == "__main__":
    main()
