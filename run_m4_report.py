"""
M4 : quantification INT8 PARTIELLE (PTQ statique, FX ; moteur fbgemm, ou qnnpack/x86 si fbgemm indisponible) d'un M3.

Ce que le rapport demande (resume) : INT8 post-entrainement, 200 images de calibration, seulement le tronc du backbone
(stem + res2..res5) et la tete de boites (box head) ; FPN, RPN et predicteur restent en float. Seule la configuration
"maximal" passe en INT8 dans le rapport.

Etapes (tout est fait dans UNE commande) :
  1. charge le M3 (float) et le mesure (latence, taille) -> reference mesuree dans la MEME session ;
  2. replie chaque FrozenBatchNorm2d dans la convolution qui la precede (exact, sans perte) : necessaire car la
     quantification FX ne sait pas fusionner un FrozenBN de Detectron2 ;
  3. calibre sur 200 images d'ENTRAINEMENT (jamais le test) ;
  4. convertit en INT8, evalue l'AP sur le jeu de test (meme protocole que M0/M3), mesure latence et taille.

Utilisation (depuis E:\\Stage2\\TinyML_Framework), UNE ligne :
  python run_m4_report.py quantize --m3 m1_report\\checkpoints\\M3_maximal_t61.7_crank_matched.pth --threads 4

Sortie : m1_report\\checkpoints\\M4_<nom>.int8.state.pt (state_dict quantifie) + M4_<nom>.quant.json.
Test rapide sans evaluation AP (quelques minutes) : ajouter --n-calib 8 --skip-eval --repeats 3

Declare : quantification PAR UNITE (stem, res2, res3, res4, res5, box head) ; les tenseurs entre unites restent en
float (une quantification/dequantification a chaque frontiere). Le state_dict INT8 n'est pas rechargeable par
evaluate_m1_report.py (modele quantifie) : l'evaluation est faite dans cette commande.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

SEED_CALIB = 1
SKIP_UNITS = set()          # unites laissees en float (option --fp32-units)


# ----------------------------------------------------------------------------- choix du moteur de quantification
def pick_backend(requested: str = "auto") -> str:
    """fbgemm (x86) si disponible ; sinon x86 / onednn ; sinon qnnpack (processeurs ARM ou builds sans fbgemm)."""
    supported = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    print(f"[M4] moteurs de quantification disponibles : {supported}", flush=True)
    order = ([requested] if requested != "auto" else []) + ["fbgemm", "x86", "onednn", "qnnpack"]
    for e in order:
        if e in supported:
            if requested != "auto" and e != requested:
                print(f"[M4] moteur demande '{requested}' indisponible, utilisation de '{e}'", flush=True)
            print(f"[M4] moteur retenu : {e}", flush=True)
            return e
    raise SystemExit(f"Aucun moteur de quantification disponible ({supported}).")


# ----------------------------------------------------------------------------- repli exact des FrozenBN
def fold_conv_bn(conv) -> nn.Conv2d:
    """detectron2 Conv2d (+ FrozenBatchNorm2d eventuel) -> nn.Conv2d ordinaire avec biais."""
    norm = getattr(conv, "norm", None)
    w = conv.weight.detach().clone()
    b = conv.bias.detach().clone() if conv.bias is not None else torch.zeros(w.shape[0])
    if norm is not None:
        scale = norm.weight * (norm.running_var + norm.eps).rsqrt()
        shift = norm.bias - norm.running_mean * scale
        w = w * scale.view(-1, 1, 1, 1)
        b = b * scale + shift
    new = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                    conv.dilation, conv.groups, bias=True)
    new.weight.data.copy_(w)
    new.bias.data.copy_(b)
    return new


class QStem(nn.Module):
    def __init__(self, stem):
        super().__init__()
        self.conv1 = fold_conv_bn(stem.conv1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.pool(self.relu(self.conv1(x)))


class QBlock(nn.Module):
    """Bottleneck equivalent (conv1-relu-conv2-relu-conv3 + raccourci, puis relu), sans FrozenBN."""

    def __init__(self, blk):
        super().__init__()
        self.conv1 = fold_conv_bn(blk.conv1)
        self.relu1 = nn.ReLU()
        self.conv2 = fold_conv_bn(blk.conv2)
        self.relu2 = nn.ReLU()
        self.conv3 = fold_conv_bn(blk.conv3)
        self.shortcut = fold_conv_bn(blk.shortcut) if blk.shortcut is not None else None
        self.relu_out = nn.ReLU()

    def forward(self, x):
        out = self.conv3(self.relu2(self.conv2(self.relu1(self.conv1(x)))))
        sc = self.shortcut(x) if self.shortcut is not None else x
        return self.relu_out(out + sc)


def fold_trunk(model):
    """Remplace stem et stages du tronc par des versions repliees (modele float, resultat identique)."""
    from detectron2.modeling.backbone.resnet import BottleneckBlock
    bu = model.backbone.bottom_up
    bu.stem = QStem(bu.stem)
    for i, stage in enumerate(bu.stages):
        for blk in stage:
            if not isinstance(blk, BottleneckBlock):
                raise SystemExit(f"Bloc non gere : {type(blk).__name__} (seul BottleneckBlock / R50 est gere)")
        bu.stages[i] = nn.Sequential(*[QBlock(b) for b in stage])
    return model


# ----------------------------------------------------------------------------- quantification FX par unite
def quant_units(model):
    """(nom, parent, attribut/index, module) des unites quantifiees : stem, res2..res5, box_head."""
    bu = model.backbone.bottom_up
    units = [("stem", bu, "stem", bu.stem)]
    for i, (name, stage) in enumerate(zip(bu.stage_names, bu.stages)):
        units.append((name, bu.stages, i, stage))
    units.append(("box_head", model.roi_heads, "box_head", model.roi_heads.box_head))
    return [u for u in units if u[0] not in SKIP_UNITS]


def _set(parent, key, new):
    if isinstance(key, int):
        parent[key] = new
    else:
        setattr(parent, key, new)


def prepare_units(model, example_by_unit, backend="fbgemm"):
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx
    torch.backends.quantized.engine = backend
    qmap = get_default_qconfig_mapping(backend)
    for name, parent, key, mod in quant_units(model):
        prepared = prepare_fx(mod.eval(), qmap, (example_by_unit[name],))
        _set(parent, key, prepared)


def convert_units(model):
    from torch.ao.quantization.quantize_fx import convert_fx
    for name, parent, key, mod in quant_units(model):
        _set(parent, key, convert_fx(mod))


def capture_unit_inputs(model, inputs):
    """Entrees reelles de chaque unite (pour tracer les formes), via des hooks sur UNE image."""
    got, hs = {}, []

    def make(n):
        def hook(m, i, o):                       # ne RIEN retourner : un retour remplacerait la sortie du module
            got.setdefault(n, i[0].detach().clone())
        return hook

    for name, parent, key, mod in quant_units(model):
        hs.append(mod.register_forward_hook(make(name)))
    with torch.no_grad():
        model(inputs)
    for h in hs:
        h.remove()
    return got


def calibrate(model, mapper, dicts):
    n = len(dicts)
    t0 = time.time()
    with torch.no_grad():
        for k, d in enumerate(dicts, 1):
            model([mapper(d)])
            if k % 20 == 0 or k == n:
                el = time.time() - t0
                print(f"  calibration {k}/{n}  (reste ~{el / k * (n - k) / 60:.1f} min)", flush=True)


def quantize_model(model, mapper, dicts, backend="fbgemm", check_input=None):
    """model : float M3. Retourne (modele_int8, infos). check_input : [dict] pour les controles numeriques."""
    infos = {}
    model.eval()
    ref = copy.deepcopy(model) if check_input is not None else None
    fold_trunk(model)
    if check_input is not None:
        with torch.no_grad():
            fa = ref.backbone(ref.preprocess_image(check_input).tensor)
            fb = model.backbone(model.preprocess_image(check_input).tensor)
        infos["fold_max_rel_diff"] = max(float((fa[k] - fb[k]).abs().max() / fa[k].abs().max().clamp_min(1e-12)) for k in fa)
    ex = capture_unit_inputs(model, [mapper(dicts[0])])
    prepare_units(model, ex, backend)
    calibrate(model, mapper, dicts)
    convert_units(model)
    if check_input is not None:
        with torch.no_grad():
            fc = model.backbone(model.preprocess_image(check_input).tensor)
        infos["fpn_cosine_vs_float"] = {
            k: float(torch.nn.functional.cosine_similarity(fa[k].flatten(), fc[k].flatten(), dim=0)) for k in fa}
    return model, infos


def state_dict_bytes(sd) -> int:
    return sum(t.numel() * t.element_size() for t in sd.values() if torch.is_tensor(t))


# ----------------------------------------------------------------------------- commande
def cmd_quantize(args):
    import run_m1_report as R
    import verify_m0_8000 as V
    import measure_costs as C
    from detectron2.data import DatasetCatalog, DatasetMapper
    from detectron2.modeling import build_model
    from evaluate_m0_m1_pruned import reconstruct_pruned_architecture

    torch.set_num_threads(args.threads)
    backend = pick_backend(args.backend)
    SKIP_UNITS.clear()
    SKIP_UNITS.update(x for x in args.fp32_units.split(",") if x)
    if SKIP_UNITS:
        print(f"[M4] unites laissees en FLOAT : {sorted(SKIP_UNITS)}", flush=True)
    m3_path = Path(args.m3)
    blob = torch.load(m3_path, map_location="cpu", weights_only=False)
    if blob.get("stage") != "M3":
        raise SystemExit("Ce fichier n'est pas un M3 (distillation). Lancez d'abord run_m3_report.py.")
    if blob["pruning_report"].get("m0_sha256") != R.M0_SHA256:
        raise SystemExit("Ce M3 ne vient pas de M0-8000 (SHA256 different). Arret.")
    base = m3_path.stem[3:] + ("_fp32_" + args.fp32_units.replace(",", "-") if args.fp32_units else "")
    out_state = R.CKPT_DIR / f"M4_{base}.int8.state.pt"
    out_json = out_state.with_name(f"M4_{base}.quant.json")

    cfg = V.build_cfg(m3_path)
    model = build_model(cfg)
    sd = blob["model_state_dict"]
    reconstruct_pruned_architecture(model, sd)
    model.load_state_dict(sd, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[M4] {m3_path.name} : {n_params:,} parametres", flush=True)

    R.register_train()
    V.register()
    dicts = [d for d in DatasetCatalog.get(R.TRAIN_NAME) if len(d.get("annotations", [])) > 0]
    random.Random(SEED_CALIB).shuffle(dicts)
    dicts = dicts[:args.n_calib]
    mapper = DatasetMapper(cfg, is_train=False)
    test_dict = DatasetCatalog.get(V.DATASET)[0]
    import cv2
    bgr = cv2.imread(test_dict["file_name"])

    print("[M4] mesure du M3 float (reference, meme session)...", flush=True)
    float_cost = C.measure(model, cfg, bgr, "M3_float", args.threads, args.repeats, sd=sd, ckpt_path=m3_path)
    check = [C.preprocess(bgr, cfg)]

    print(f"[M4] repli des FrozenBN + calibration sur {len(dicts)} images d'entrainement...", flush=True)
    qmodel, infos = quantize_model(model, mapper, dicts, backend=backend, check_input=check)

    qsd = qmodel.state_dict()
    R.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(qsd, out_state)
    gates = {"fold_exact": infos.get("fold_max_rel_diff", 1.0) < 1e-4,
             "fpn_cosine_above_0.9": all(v > 0.9 for v in infos["fpn_cosine_vs_float"].values())}
    with torch.no_grad():
        o = qmodel([C.preprocess(bgr, cfg)])[0]["instances"]
    gates["forward_finite"] = bool(torch.isfinite(o.scores).all()) if len(o) else True
    gates["n_detections_at_0.05_on_one_image"] = int(len(o))

    q_cost = C.measure(qmodel, cfg, bgr, "M4_int8", args.threads, args.repeats)   # latence (flops non comptes en int8)
    q_cost.pop("flops_error", None)
    q_cost["parameters_float_equivalent"] = n_params
    q_cost["size_mb_state_dict"] = state_dict_bytes(qsd) / 1e6
    q_cost["size_mb_file"] = out_state.stat().st_size / 1e6
    q_cost["GFLOPs_note"] = "identique au M3 float (la quantification ne change pas le nombre d'operations)"

    res = {}
    if not args.skip_eval:
        print("[M4] evaluation sur le jeu de test (peut prendre ~20-40 min)...", flush=True)
        res = V.run(qmodel, cfg, out_state.with_name(f"M4_{base}_eval"))
        m0 = json.loads(V.DEFAULT_OUT.joinpath("verify_m0_8000.json").read_text(encoding="utf-8"))
        res["retention_AP"] = res["AP"] / m0["AP"]
        res["retention_AP50"] = res["AP50"] / m0["AP50"]
    report = {"source_m3": str(m3_path), "n_calib": len(dicts), "calib_seed": SEED_CALIB, "backend": backend,
              "scope": "stem + res2..res5 + box_head en INT8 ; FPN, RPN, predicteur en float",
              "granularity": "par unite (frontieres en float)", "infos": infos, "gates": gates,
              "float_m3": float_cost, "int8": q_cost, "eval": res,
              "stage": "M4", "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"M4 [{base}]")
    print("PORTES :", json.dumps(gates, indent=1))
    print(f"latence mediane   float : {float_cost['latency']['median_ms']:.0f} ms   int8 : {q_cost['latency']['median_ms']:.0f} ms")
    print(f"taille state_dict float : {float_cost['size_mb_state_dict']:.1f} Mo   int8 : {q_cost['size_mb_state_dict']:.1f} Mo")
    if res:
        print(f"AP {res['AP']:.2f}  AP50 {res['AP50']:.2f}  AP75 {res['AP75']:.2f}   retention_AP {res['retention_AP']:.4f}")
    print("ecrit :", out_json)
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quantize")
    q.add_argument("--m3", required=True)
    q.add_argument("--n-calib", type=int, default=200)
    q.add_argument("--threads", type=int, default=4)
    q.add_argument("--repeats", type=int, default=10)
    q.add_argument("--skip-eval", action="store_true")
    q.add_argument("--fp32-units", default="", help="unites gardees en float : stem,res2,res3,res4,res5,box_head")
    q.add_argument("--backend", default="auto", help="auto (defaut), fbgemm, x86, onednn ou qnnpack")
    args = ap.parse_args()
    {"quantize": cmd_quantize}[args.cmd](args)


if __name__ == "__main__":
    main()