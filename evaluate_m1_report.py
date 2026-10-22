"""
Evaluation d'un checkpoint M1 (ou plus tard M2/M3) avec le MEME protocole que M0-8000 :
AP COCO a seuil 0.05, TP/FP/FN a 0.5 avec la vraie verite terrain, puis RETENTION = AP / AP(M0-8000).

Utilisation (depuis E:\\Stage2\\TinyML_Framework, verify_m0_8000.py dans le meme dossier) :
    python evaluate_m1_report.py --checkpoint m1_report\\checkpoints\\M1R_expose_whole_backbone_r0.55.pth

Rappel : dans le rapport, M1 SEUL s'effondre (AP ~ 0 a 0.3). Ce n'est pas un echec : la recuperation
(M2 recalibration, puis M3 distillation) est ce qui redonne l'AP. Evaluer M1 seul est surtout un controle
que le modele charge et predit sans erreur.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import verify_m0_8000 as V
from evaluate_m0_m1_pruned import reconstruct_pruned_architecture   # reconstruction deja validee
from detectron2.modeling import build_model

M0_JSON = V.DEFAULT_OUT / "verify_m0_8000.json"


def state_dict_mb(sd) -> float:
    return sum(t.numel() * t.element_size() for t in sd.values() if torch.is_tensor(t)) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ckpt = Path(args.checkpoint)
    out = Path(args.out) if args.out else ckpt.parent / (ckpt.stem + "_eval")

    V.register()
    cfg = V.build_cfg(ckpt)
    model = build_model(cfg)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = blob["model_state_dict"]
    reconstruct_pruned_architecture(model, sd)
    model.load_state_dict(sd, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] parametres = {n_params:,}   taille (state_dict) = {state_dict_mb(sd):.1f} MB", flush=True)

    res = V.run(model, cfg, out)
    # Un modele qui ne produit AUCUNE detection (M1 effondre) donne un AP COCO indefini (nan ou -1).
    # Par definition ce modele a AP = 0 : on le force a 0 pour que la retention soit calculable.
    for k in ("AP", "AP50", "AP75", "APs", "APm", "APl"):
        v = res.get(k)
        if v is not None and (not isinstance(v, (int, float)) or math.isnan(v) or v < 0):
            res[k] = 0.0
            res.setdefault("note", "aucune detection -> AP indefini, fixe a 0")
    res.update({"checkpoint": str(ckpt), "parameters": n_params, "size_mb_state_dict": state_dict_mb(sd)})
    if M0_JSON.is_file():
        m0 = json.loads(M0_JSON.read_text(encoding="utf-8"))
        res["retention_AP"] = res["AP"] / m0["AP"]
        res["retention_AP50"] = res["AP50"] / m0["AP50"]
        res["params_reduction_pct"] = 100.0 * (m0["parameters"] - n_params) / m0["parameters"]
    (out / "eval.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 60)
    for k in ("AP", "AP50", "AP75", "Precision", "Recall", "F1", "Jaccard", "parameters",
              "size_mb_state_dict", "params_reduction_pct", "retention_AP", "retention_AP50"):
        if k in res:
            print(f"{k:22} {res[k]}")
    print("=" * 60)


if __name__ == "__main__":
    main()