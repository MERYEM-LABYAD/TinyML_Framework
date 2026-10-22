"""
Mesure du COUT des modeles (parametres, taille, GFLOPs, latence CPU) avec un protocole UNIQUE et declare.

Pourquoi un protocole declare ? Les formules existent (compter les multiplications-accumulations des couches),
mais le resultat depend de CHOIX que le rapport ne precise pas : taille d'entree, MAC ou FLOPs (1 MAC = 2 FLOPs),
nombre de propositions RPN, nombre de threads, echauffement. On les fixe et on les ecrit dans le rapport.

Protocole (valeurs par defaut) :
  - entree : une vraie image du jeu de test, redimensionnee comme a l'evaluation (plus petit cote 800 px, max 1333) ;
  - GFLOPs : fvcore (via detectron2.utils.analysis.FlopCountAnalysis) = conv + lineaire + matmul ; on donne
             les GMACs (ce que fvcore compte) ET GFLOPs = 2 x GMACs ;
  - latence : CPU, batch 1, --threads threads, 3 passes d'echauffement puis --repeats passes mesurees, mediane (ms) ;
  - taille : octets du state_dict en float32 (Mo = 1e6 octets) et taille du fichier .pth.

Utilisation (depuis E:\\Stage2\\TinyML_Framework) :
    python measure_costs.py --threads 4 --repeats 10 ^
        --ckpt M0=default ^
        --ckpt M3_whole=m1_report\\checkpoints\\M3_expose_whole_backbone_t41.8_crank_matched.pth ^
        --ckpt M3_aggr=m1_report\\checkpoints\\M3_aggressive_t57.6_crank_matched.pth ^
        --ckpt M3_max=m1_report\\checkpoints\\M3_maximal_t61.7_crank_matched.pth
(en PowerShell, ecrire tout sur UNE ligne, sans les ^). Sortie : m1_report\\costs.json + tableau.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch


def state_dict_bytes(sd) -> int:
    return sum(t.numel() * t.element_size() for t in sd.values() if torch.is_tensor(t))


def preprocess(bgr, cfg):
    from detectron2.data import transforms as T
    aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
    img = aug.get_transform(bgr).apply_image(bgr)
    return {"image": torch.as_tensor(img.astype("float32").transpose(2, 0, 1)),
            "height": bgr.shape[0], "width": bgr.shape[1]}


def count_gmacs(model, inputs) -> float:
    """GMACs comptes par fvcore (conv, lineaire, matmul). Les operations sans parametres (NMS, ...) sont ignorees."""
    from detectron2.utils.analysis import FlopCountAnalysis
    fa = FlopCountAnalysis(model, inputs)
    fa.unsupported_ops_warnings(False)
    fa.uncalled_modules_warnings(False)
    return fa.total() / 1e9


def latency_ms(model, inputs, threads: int, repeats: int, warmup: int = 3):
    torch.set_num_threads(threads)
    ts = []
    with torch.no_grad():
        for i in range(warmup + repeats):
            t0 = time.perf_counter()
            model(inputs)
            dt = (time.perf_counter() - t0) * 1000
            if i >= warmup:
                ts.append(dt)
    return {"median_ms": statistics.median(ts), "mean_ms": statistics.mean(ts),
            "std_ms": statistics.pstdev(ts), "n": len(ts), "threads": threads}


def measure(model, cfg, bgr, label, threads, repeats, sd=None, ckpt_path=None):
    inputs = [preprocess(bgr, cfg)]
    h, w = inputs[0]["image"].shape[1:]
    n_params = sum(p.numel() for p in model.parameters())
    out = {"label": label, "input_hw": [int(h), int(w)], "parameters": n_params}
    if sd is not None:
        out["size_mb_state_dict"] = state_dict_bytes(sd) / 1e6
    if ckpt_path is not None and Path(ckpt_path).is_file():
        out["size_mb_file"] = Path(ckpt_path).stat().st_size / 1e6
    try:
        g = count_gmacs(model, inputs)
        out["GMACs"], out["GFLOPs"] = g, 2 * g
    except Exception as e:                                   # ne doit pas bloquer la latence
        out["flops_error"] = f"{type(e).__name__}: {e}"
    out["latency"] = latency_ms(model, inputs, threads, repeats)
    return out


def load_for_eval(spec: str):
    import verify_m0_8000 as V
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from evaluate_m0_m1_pruned import reconstruct_pruned_architecture
    path = V.DEFAULT_CHECKPOINT if spec == "default" else Path(spec)
    cfg = V.build_cfg(path)
    m = build_model(cfg)
    if spec == "default":
        DetectionCheckpointer(m).load(str(path))
        sd = m.state_dict()
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
        reconstruct_pruned_architecture(m, sd)
        m.load_state_dict(sd, strict=True)
    m.eval()
    return m, cfg, sd, path


def first_test_image():
    import cv2
    import verify_m0_8000 as V
    from detectron2.data import DatasetCatalog
    V.register()
    d = DatasetCatalog.get(V.DATASET)[0]
    return cv2.imread(d["file_name"])


def print_table(rows):
    base = rows[0]
    print("\n" + "=" * 100)
    print(f"{'modele':14}{'params (M)':>12}{'taille (Mo)':>13}{'GMACs':>10}{'GFLOPs':>10}{'latence (ms)':>14}"
          f"{'dGFLOPs':>10}{'dlatence':>10}")
    for r in rows:
        gf = r.get("GFLOPs")
        d1 = f"{100 * (gf / base['GFLOPs'] - 1):+.1f}%" if gf and base.get("GFLOPs") else "-"
        d2 = f"{100 * (r['latency']['median_ms'] / base['latency']['median_ms'] - 1):+.1f}%"
        print(f"{r['label']:14}{r['parameters'] / 1e6:12.2f}{r.get('size_mb_state_dict', float('nan')):13.1f}"
              f"{r.get('GMACs', float('nan')):10.1f}{gf if gf else float('nan'):10.1f}"
              f"{r['latency']['median_ms']:14.0f}{d1:>10}{d2:>10}")
    print("=" * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="etiquette=chemin (ou etiquette=default pour M0)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", default="m1_report/costs.json")
    args = ap.parse_args()

    bgr = first_test_image()
    rows = []
    for item in args.ckpt:
        label, spec = item.split("=", 1)
        print(f"[mesure] {label}", flush=True)
        m, cfg, sd, path = load_for_eval(spec)
        rows.append(measure(m, cfg, bgr, label, args.threads, args.repeats, sd=sd,
                            ckpt_path=None if spec == "default" else path))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"protocol": {"input": "plus petit cote 800 px, max 1333", "threads": args.threads,
                                                       "repeats": args.repeats, "warmup": 3, "batch": 1,
                                                       "flops_tool": "fvcore FlopCountAnalysis (1 MAC = 1 'flop' fvcore ; GFLOPs = 2 x GMACs)"},
                                          "rows": rows}, indent=2), encoding="utf-8")
    print_table(rows)
    print("ecrit :", args.out)


if __name__ == "__main__":
    main()
