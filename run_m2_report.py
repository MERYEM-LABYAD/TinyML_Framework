"""
M2 conforme au rapport : RECALIBRATION des statistiques de normalisation (FrozenBN), a partir d'un M1.

Rapport, 4.6 : "Recalibration resets the statistics and recomputes them by running a few hundred
calibration images through the pruned model in forward-only mode, then re-freezes them. [...] for a
detector with frozen batch-normalization, fine-tuning cannot update the statistics at all, so explicit
recalibration is mandatory."

POINT TECHNIQUE (a lire) : dans le rapport (torchvision), les FrozenBN de M0 portent de VRAIES statistiques
ImageNet ; recalibrer = remplacer ces statistiques par celles du modele elague. Dans Detectron2, les
FrozenBN de M0 peuvent porter des statistiques triviales (moyenne 0, variance 1 : affine "replie"). Appliquer
alors la recalibration litterale (normaliser chaque canal par ses statistiques de donnees) RESCALERAIT chaque
canal et deformerait meme M0. On implemente donc la version GENERALE, qui se reduit exactement a la version
litterale quand M0 est bien normalise :

   pour chaque canal c de M1 (correspondant au canal c0 de M0), avec  mu0, sig0 = statistiques des
   ENTREES du BN dans M0 (donnees reelles),  mT, sT = statistiques stockees dans le BN de M0,
   mu1, sig1 = statistiques des entrees dans M1 :
        r        = sig1 / sig0
        var_new  = (r * sT)^2 - eps
        mean_new = mu1 - r * (mu0 - mT)
   -> la sortie du BN de M1 a la meme moyenne et le meme ecart-type que celle de M0 (affine gamma/beta inchangees).
   Si M0 est bien normalise (mu0 = mT, sig0 = sT) on retrouve mean_new = mu1, var_new = sig1^2 (litteral).
   Si M1 = M0 (rien elague), mean_new = mT et var_new = sT^2 - eps : IDENTITE (verifiee par 'selftest').

Mode 'plain' : version litterale (mean_new = mu1, var_new = sig1^2).

Commandes (depuis E:\\Stage2\\TinyML_Framework) :
  python run_m2_report.py inspect                         # statistiques des FrozenBN de M0-8000 (rapide)
  python run_m2_report.py selftest                        # M0 recalibre sur lui-meme doit rester identique
  python run_m2_report.py recalibrate --m1 m1_report\\checkpoints\\M1R_expose_whole_backbone_t41.8_crank.pth
  python evaluate_m1_report.py --checkpoint m1_report\\checkpoints\\M2_<...>.pth     # AP / AP50 / AP75 de M2
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_m1_report as R                                   # noqa: E402

SEED_RECAL = 1          # images de calibration : graine differente de celle du calcul Taylor (graine 0)


# ----------------------------------------------------------------- utilitaires
def frozen_bns(model):
    from detectron2.layers import FrozenBatchNorm2d
    return {n: m for n, m in model.named_modules() if isinstance(m, FrozenBatchNorm2d)}


def calib_dicts(n):
    from detectron2.data import DatasetCatalog
    dicts = [d for d in DatasetCatalog.get(R.TRAIN_NAME) if len(d.get("annotations", [])) > 0]
    random.Random(SEED_RECAL).shuffle(dicts)
    return dicts[:n]


@torch.no_grad()
def run_backbone(model, mapper, dicts, bns, hook_factory, tag):
    """Passe avant du BACKBONE seul (les FrozenBN n'existent que la), une image a la fois."""
    handles = [bn.register_forward_hook(hook_factory(name)) for name, bn in bns.items()]
    model.eval()
    t0 = time.time()
    try:
        for i, d in enumerate(dicts):
            batch = [mapper(copy.deepcopy(d))]
            images = model.preprocess_image(batch)
            model.backbone(images.tensor)
            if (i + 1) % 10 == 0 or i + 1 == len(dicts):
                print(f"  [{tag}] {i + 1}/{len(dicts)}  ({(time.time() - t0) / (i + 1):.1f} s/image)", flush=True)
    finally:
        for h in handles:
            h.remove()


class Acc:
    """Somme / somme des carres par canal des ENTREES d'un BN (float64)."""

    def __init__(self, c):
        self.s = torch.zeros(c, dtype=torch.float64)
        self.q = torch.zeros(c, dtype=torch.float64)
        self.n = 0

    def add(self, x):
        self.s += x.sum(dim=(0, 2, 3), dtype=torch.float64)
        self.q += (x * x).sum(dim=(0, 2, 3), dtype=torch.float64)
        self.n += x.shape[0] * x.shape[2] * x.shape[3]

    def mean_std(self):
        mu = self.s / self.n
        var = (self.q / self.n - mu * mu).clamp_min(1e-12)
        return mu, var.sqrt()


def collect_input_stats(model, mapper, dicts, tag):
    """Statistiques des entrees de chaque FrozenBN avec les statistiques ACTUELLES du modele."""
    bns = frozen_bns(model)
    acc = {n: Acc(int(bn.running_mean.numel())) for n, bn in bns.items()}

    def factory(name):
        def hook(module, inputs, output):
            acc[name].add(inputs[0])
        return hook

    run_backbone(model, mapper, dicts, bns, factory, tag)
    return {n: a.mean_std() for n, a in acc.items()}


def collect_batchstats_init(student, mapper, dicts, teacher_ref, kept, mode, tag):
    """
    Initialisation 'Precise-BN' : le BN normalise par les statistiques de l'image courante (comme un BN
    en entrainement), puis remet l'echelle de M0 (matched) ou non (plain), pendant que l'on accumule les
    statistiques des entrees pour estimer mu1, sig1.
    """
    bns = frozen_bns(student)
    acc = {n: Acc(int(bn.running_mean.numel())) for n, bn in bns.items()}
    ref = {}
    for n, bn in bns.items():
        conv = n[: -len(".norm")]
        k = kept[conv]
        (mu0, sig0), (mT, sT) = teacher_ref[n]
        mu0, sig0, mT, sT = mu0[k], sig0[k], mT[k], sT[k]
        if mode == "plain":
            zscale, zshift = torch.ones_like(sig0), torch.zeros_like(sig0)
        else:
            zscale, zshift = sig0 / sT, (mu0 - mT) / sT
        ref[n] = (zscale.float().view(1, -1, 1, 1), zshift.float().view(1, -1, 1, 1))

    def factory(name):
        bn = bns[name]
        zscale, zshift = ref[name]

        def hook(module, inputs, output):
            x = inputs[0]
            acc[name].add(x)
            mb = x.mean(dim=(0, 2, 3), keepdim=True)
            vb = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            z = (x - mb) / (vb + 1e-12).sqrt() * zscale + zshift
            return z * bn.weight.view(1, -1, 1, 1) + bn.bias.view(1, -1, 1, 1)
        return hook

    run_backbone(student, mapper, dicts, bns, factory, tag)
    return {n: a.mean_std() for n, a in acc.items()}


def apply_buffers(student, stats1, teacher_ref, kept, mode, only=None):
    """Ecrit running_mean / running_var dans les FrozenBN de l'etudiant. Retourne des compteurs de controle."""
    bns = frozen_bns(student)
    clamped = 0
    total = 0
    for n, bn in bns.items():
        if only is not None and n not in only:
            continue
        conv = n[: -len(".norm")]
        k = kept[conv]
        (mu0, sig0), (mT, sT) = teacher_ref[n]
        mu0, sig0, mT, sT = mu0[k], sig0[k], mT[k], sT[k]
        mu1, sig1 = stats1[n]
        if mode == "plain":
            mean_new, s_new = mu1, sig1
        else:
            r = (sig1 / sig0.clamp_min(1e-6)).clamp(1e-3, 1e3)
            mean_new = mu1 - r * (mu0 - mT)
            s_new = r * sT
        var_new = s_new * s_new - bn.eps
        clamped += int((var_new < 1e-8).sum())
        total += var_new.numel()
        with torch.no_grad():
            bn.running_mean.copy_(mean_new.to(bn.running_mean.dtype))
            bn.running_var.copy_(var_new.clamp_min(1e-8).to(bn.running_var.dtype))
    return {"channels_total": total, "var_clamped": clamped}


class _Stop(Exception):
    """Arret anticipe de la passe avant du backbone (on n'a plus besoin de la suite)."""


def bn_groups(student):
    """FrozenBN regroupes dans l'ordre de la passe avant : stem, puis un groupe par bloc residuel."""
    names = list(frozen_bns(student).keys())
    groups, index = [], {}
    for n in names:
        key = "stem" if ".stem." in n else ".".join(n.split(".")[:4])      # backbone.bottom_up.resK.B
        if key not in index:
            index[key] = len(groups)
            groups.append((key, []))
        groups[index[key]][1].append(n)
    return groups


def calibrate_group(student, mapper, dicts, group, teacher_ref, kept, mode, tag, frozen=False):
    """
    Passe avant jusqu'au DERNIER BN du groupe seulement (arret anticipe). Les BN des groupes precedents ont deja
    leurs statistiques FINALES. Les BN du groupe normalisent a la volee (statistiques de l'image courante, remises a
    l'echelle de M0), pendant que l'on accumule les statistiques de leurs entrees.
    """
    bns_all = frozen_bns(student)
    bns = {n: bns_all[n] for n in group}
    acc = {n: Acc(int(bn.running_mean.numel())) for n, bn in bns.items()}
    ref = {}
    for n, bn in bns.items():
        k = kept[n[: -len(".norm")]]
        (mu0, sig0), (mT, sT) = teacher_ref[n]
        mu0, sig0, mT, sT = mu0[k], sig0[k], mT[k], sT[k]
        if mode == "plain":
            zscale, zshift = torch.ones_like(sig0), torch.zeros_like(sig0)
        else:
            zscale, zshift = sig0 / sT, (mu0 - mT) / sT
        ref[n] = (zscale.float().view(1, -1, 1, 1), zshift.float().view(1, -1, 1, 1))
    seen = set()
    order = list(group)

    def factory(name):
        bn = bns[name]
        zscale, zshift = ref[name]

        def hook(module, inputs, output):
            x = inputs[0]
            acc[name].add(x)
            seen.add(name)
            if frozen:                      # tours suivants : le BN garde ses statistiques ACTUELLES (rien a la volee)
                if len(seen) == len(order):
                    raise _Stop()
                return None
            mb = x.mean(dim=(0, 2, 3), keepdim=True)
            vb = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            z = (x - mb) / (vb + 1e-12).sqrt() * zscale + zshift
            y = z * bn.weight.view(1, -1, 1, 1) + bn.bias.view(1, -1, 1, 1)
            if len(seen) == len(order):
                raise _Stop()
            return y
        return hook

    handles = [bn.register_forward_hook(factory(n)) for n, bn in bns.items()]
    student.eval()
    t0 = time.time()
    try:
        with torch.no_grad():
            for i, d in enumerate(dicts):
                seen.clear()
                batch = [mapper(copy.deepcopy(d))]
                images = student.preprocess_image(batch)
                try:
                    student.backbone(images.tensor)
                except _Stop:
                    pass
    finally:
        for h in handles:
            h.remove()
    print(f"  [{tag}] {len(dicts)} images en {time.time() - t0:.0f} s", flush=True)
    return {n: a.mean_std() for n, a in acc.items()}


def sequential_calibration(student, mapper, dicts, teacher_ref, kept, mode, rounds=2):
    """
    Calibration dans l'ordre des couches (Gauss-Seidel). Tour 1 : chaque groupe normalise a la volee (statistiques de
    l'image) pendant qu'on mesure ses entrees. Tours suivants : les BN gardent leurs statistiques actuelles (donnees
    a l'echelle du jeu d'images, pas de l'image), on re-mesure groupe par groupe et on remplace. La derive entre
    tours doit tendre vers 0 : c'est le critere de convergence.
    """
    groups = bn_groups(student)
    counters = {"channels_total": 0, "var_clamped": 0}
    prev_sig = {}
    round_drifts = []
    for rd in range(rounds):
        cur_sig = {}
        counters = {"channels_total": 0, "var_clamped": 0}
        for gi, (key, names) in enumerate(groups):
            st = calibrate_group(student, mapper, dicts, names, teacher_ref, kept, mode,
                                 f"tour {rd + 1}/{rounds} groupe {gi + 1}/{len(groups)} {key}", frozen=(rd > 0))
            c = apply_buffers(student, st, teacher_ref, kept, mode, only=set(names))
            counters["channels_total"] += c["channels_total"]
            counters["var_clamped"] += c["var_clamped"]
            for n in names:
                cur_sig[n] = st[n][1]
        if prev_sig:
            d = torch.cat([(cur_sig[n] / prev_sig[n].clamp_min(1e-12)).log().abs() for n in cur_sig])
            round_drifts.append((float(d.median()), float(d.quantile(0.95))))
            print(f"  derive tour {rd}->{rd + 1} : mediane {round_drifts[-1][0]:.4f}  p95 {round_drifts[-1][1]:.4f}",
                  flush=True)
        prev_sig = cur_sig
    counters["round_drifts"] = round_drifts
    return counters


def teacher_reference(teacher, mapper, dicts):
    """(mu0, sig0) = statistiques DONNEES des entrees ; (mT, sT) = statistiques stockees dans le BN de M0."""
    data = collect_input_stats(teacher, mapper, dicts, "M0")
    ref = {}
    for n, bn in frozen_bns(teacher).items():
        mT = bn.running_mean.detach().double().cpu()
        sT = (bn.running_var.detach().double().cpu() + bn.eps).sqrt()
        ref[n] = (data[n], (mT, sT))
    return ref


def drift(stats_a, stats_b):
    """Ecart median |log(sig_b / sig_a)| entre deux estimations des ecarts-types d'entree (convergence)."""
    d = torch.cat([(stats_b[n][1] / stats_a[n][1].clamp_min(1e-12)).log().abs() for n in stats_a])
    return float(d.median()), float(d.quantile(0.95))


def match_report(student, teacher_ref, kept, mapper, dicts):
    """Controle : sous les statistiques finales, la sortie de chaque BN a-t-elle la moyenne et l'ecart-type de M0 ?"""
    stats1 = collect_input_stats(student, mapper, dicts, "verif")
    ratios, shifts = [], []
    for n, bn in frozen_bns(student).items():
        conv = n[: -len(".norm")]
        k = kept[conv]
        (mu0, sig0), (mT, sT) = teacher_ref[n]
        mu0, sig0, mT, sT = mu0[k], sig0[k], mT[k], sT[k]
        g = bn.weight.detach().double().cpu()
        b = bn.bias.detach().double().cpu()
        mu1, sig1 = stats1[n]
        m1 = bn.running_mean.detach().double().cpu()
        s1 = (bn.running_var.detach().double().cpu() + bn.eps).sqrt()
        std_y1 = g.abs() * sig1 / s1
        std_y0 = g.abs() * sig0 / sT
        mean_y1 = g * (mu1 - m1) / s1 + b
        mean_y0 = g * (mu0 - mT) / sT + b
        ok = std_y0 > 1e-8
        ratios.append((std_y1[ok] / std_y0[ok]))
        shifts.append(((mean_y1 - mean_y0)[ok] / std_y0[ok]))
    ratios = torch.cat(ratios)
    shifts = torch.cat(shifts)
    return {
        "std_ratio_median": float(ratios.median()),
        "std_ratio_p05": float(ratios.quantile(0.05)),
        "std_ratio_p95": float(ratios.quantile(0.95)),
        "mean_shift_abs_median": float(shifts.abs().median()),
        "mean_shift_abs_p95": float(shifts.abs().quantile(0.95)),
    }


# ----------------------------------------------------------------- canaux conserves par M1
def kept_from_report(report, teacher):
    """Recalcule (de facon deterministe) quels canaux de M0 ont ete gardes dans ce M1."""
    from tinyml.pruning.report_policy import CONFIGS

    mode = report.get("mode", "channels")
    pol = copy.deepcopy(CONFIGS[report["config"]])
    pol.clip = report.get("percentile_clip", pol.clip)
    if mode == "zeros":
        from tinyml.importance.report_zeros import (channel_scores_from_weights, normalize_groups,
                                                    plan_zero_removals)
        wblob = R.load_weight_scores()
        pr = R.make_pruner(teacher, channel_scores_from_weights(wblob["scores"]))
        norm = normalize_groups(pr.units, wblob["scores"], report["group"], pol.clip, pol.min_keep)
        plan, _, _ = plan_zero_removals(pr.units, norm, pol.weights, report["ratio_used"],
                                        pol.filter_zero_thresh, pol.min_keep)
    else:
        blob = R.load_scores()
        pr = R.make_pruner(teacher, blob["scores"])
        plan, _ = pr.plan(pol, report["ratio_used"])
    # controle : meme nombre de canaux retires par unite que dans le checkpoint
    by_uid = {row["uid"]: row for row in report["channel_map"]}
    for u in pr.units:
        want = int(by_uid[u.uid]["removed"])
        got = len(plan.get(u.uid, []))
        if want != got:
            raise RuntimeError(f"Plan recalcule different du checkpoint (unite {u.uid} {u.members[:1]}): "
                               f"{got} != {want}. Les scores / parametres ont change ; arret.")
    kept = {}
    for u in pr.units:
        removed = set(plan.get(u.uid, []))
        k = torch.tensor([i for i in range(u.n_channels) if i not in removed], dtype=torch.long)
        for m in u.members:
            kept[m] = k
    return kept


# ----------------------------------------------------------------- commandes
def cmd_inspect(args):
    torch.set_num_threads(args.threads)
    cfg = R.build_cfg()
    R.register_train()
    m0 = R.load_m0(cfg)
    bns = frozen_bns(m0)
    print(f"\n{len(bns)} FrozenBN dans M0-8000")
    mean_abs, vmin, vmax = [], [], []
    for n, bn in bns.items():
        mean_abs.append(float(bn.running_mean.abs().max()))
        vmin.append(float(bn.running_var.min()))
        vmax.append(float(bn.running_var.max()))
    print(f"max |running_mean| par couche : min {min(mean_abs):.4g}  max {max(mean_abs):.4g}")
    print(f"running_var                   : min {min(vmin):.4g}  max {max(vmax):.4g}")
    trivial = max(mean_abs) < 1e-6 and abs(min(vmin) - 1.0) < 1e-3 and abs(max(vmax) - 1.0) < 1e-3
    first = next(iter(bns.items()))
    print(f"exemple {first[0]} : eps={first[1].eps}, weight[:4]={first[1].weight[:4].tolist()}, "
          f"bias[:4]={first[1].bias[:4].tolist()}")
    print("\nCONCLUSION :", "statistiques TRIVIALES (0 / 1) : BN = simple affine. La recalibration litterale "
          "rescalerait les canaux ; utiliser le mode 'matched' (defaut)." if trivial else
          "statistiques NON triviales (vraies statistiques) : 'plain' et 'matched' sont proches.")


def cmd_selftest(args):
    """M0 recalibre sur lui-meme (rien d'elague) : les statistiques doivent rester identiques."""
    torch.set_num_threads(args.threads)
    from detectron2.data import DatasetMapper
    cfg = R.build_cfg()
    R.register_train()
    teacher = R.load_m0(cfg)
    student = copy.deepcopy(teacher)
    mapper = DatasetMapper(cfg, is_train=False)
    dicts = calib_dicts(args.n_images)
    kept = {n[: -len(".norm")]: torch.arange(int(bn.running_mean.numel())) for n, bn in frozen_bns(teacher).items()}
    ref = teacher_reference(teacher, mapper, dicts)
    before = {n: (bn.running_mean.clone(), bn.running_var.clone()) for n, bn in frozen_bns(student).items()}
    stats1 = collect_input_stats(student, mapper, dicts, "selftest")
    apply_buffers(student, stats1, ref, kept, "matched")
    dm = max(float((bn.running_mean - before[n][0]).abs().max()) for n, bn in frozen_bns(student).items())
    dv = max(float((bn.running_var - before[n][1]).abs().max()) for n, bn in frozen_bns(student).items())
    print(f"\nSELFTEST : ecart max running_mean = {dm:.3g}, running_var = {dv:.3g}")
    print("RESULTAT :", "OK identite (recalibration neutre sur un modele non elague)" if dm < 1e-3 and dv < 1e-3
          else "ECART : envoyer ce journal")


def cmd_recalibrate(args):
    from detectron2.data import DatasetMapper
    from detectron2.modeling import build_model
    from evaluate_m0_m1_pruned import reconstruct_pruned_architecture

    torch.set_num_threads(args.threads)
    m1_path = Path(args.m1)
    cfg = R.build_cfg()
    R.register_train()
    blob = torch.load(m1_path, map_location="cpu", weights_only=False)
    report = blob["pruning_report"]
    if report.get("m0_sha256") != R.M0_SHA256:
        raise SystemExit("Ce M1 ne vient pas de M0-8000 (SHA256 different). Arret.")
    print(f"[M1] {m1_path.name} : config={report['config']} mode={report.get('mode', 'channels')} "
          f"reduction={report['reduction_pct']:.2f} %", flush=True)

    teacher = R.load_m0(cfg)
    kept = kept_from_report(report, teacher)

    student = build_model(cfg)
    sd = blob["model_state_dict"]
    reconstruct_pruned_architecture(student, sd)
    student.load_state_dict(sd, strict=True)
    n_params = sum(p.numel() for p in student.parameters())
    bns = frozen_bns(student)
    for n, bn in bns.items():
        conv = n[: -len(".norm")]
        if conv not in kept or int(kept[conv].numel()) != int(bn.running_mean.numel()):
            raise RuntimeError(f"Correspondance de canaux impossible pour {n}")
    print(f"[M2] {len(bns)} FrozenBN a recalibrer ; parametres = {n_params:,}", flush=True)

    mapper = DatasetMapper(cfg, is_train=False)
    dicts = calib_dicts(args.n_images)
    print(f"[M2] {len(dicts)} images de calibration (graine {SEED_RECAL}), mode={args.mode}", flush=True)

    print("\n== 1/4  statistiques de M0 (reference) ==", flush=True)
    cache = R.OUT / f"m2_teacher_ref_n{len(dicts)}_seed{SEED_RECAL}.pt"
    ref = None
    if cache.is_file():
        c = torch.load(cache, map_location="cpu", weights_only=False)
        if c.get("m0_sha") == R.M0_SHA256 and c.get("n") == len(dicts):
            ref = c["ref"]
            print(f"  reference M0 relue depuis {cache.name} (meme M0, memes images) : gain de temps", flush=True)
    if ref is None:
        ref = teacher_reference(teacher, mapper, dicts)
        torch.save({"ref": ref, "m0_sha": R.M0_SHA256, "n": len(dicts)}, cache)

    if args.init == "sequential":
        print("\n== 2/4  calibration sequentielle (groupe par groupe, dans l'ordre des couches) ==", flush=True)
        seq_dicts = dicts[: args.n_seq_images]
        counters = sequential_calibration(student, mapper, seq_dicts, ref, kept, args.mode, rounds=args.seq_rounds)
    else:
        print("\n== 2/4  initialisation globale (statistiques par image) ==", flush=True)
        stats1 = collect_batchstats_init(student, mapper, dicts, ref, kept, args.mode, "init")
        counters = apply_buffers(student, stats1, ref, kept, args.mode)

    refine_dicts = dicts[: args.n_refine_images]
    drifts = []
    prev = None
    for it in range(args.refine):
        print(f"\n== 3/4  raffinement {it + 1}/{args.refine} (statistiques gelees, {len(refine_dicts)} images) ==",
              flush=True)
        stats1 = collect_input_stats(student, mapper, refine_dicts, f"refine{it + 1}")
        if prev is not None:
            drifts.append(drift(prev, stats1))
            print(f"  derive entre passes : mediane {drifts[-1][0]:.4f}  p95 {drifts[-1][1]:.4f}", flush=True)
        prev = stats1
        counters = apply_buffers(student, stats1, ref, kept, args.mode)

    print("\n== 4/4  controle d'appariement avec M0 ==", flush=True)
    check = match_report(student, ref, kept, mapper, dicts[-args.n_verify:])   # images DIFFERENTES de la calibration

    gates = {}
    finite = all(torch.isfinite(b).all().item() for bn in bns.values() for b in (bn.running_mean, bn.running_var))
    gates["bn_count_53"] = (len(bns) == 53)
    gates["buffers_finite"] = bool(finite)
    gates["running_var_positive"] = all(bool((bn.running_var > 0).all()) for bn in bns.values())
    gates["var_clamped_fraction_below_1pct"] = counters["var_clamped"] <= 0.01 * counters["channels_total"]
    gates["params_unchanged"] = (n_params == report["params_after"])
    if args.mode == "matched":
        gates["output_std_matches_M0"] = 0.8 <= check["std_ratio_median"] <= 1.25
    try:
        with torch.no_grad():
            student.eval()
            out = student(R.real_image_input())
        inst = out[0]["instances"]
        gates["forward_finite"] = all(torch.isfinite(v).all().item()
                                      for v in inst.get_fields().values() if torch.is_tensor(v))
        gates["n_detections_at_0.05_on_one_image"] = int(len(inst))
    except Exception as exc:                                  # noqa: BLE001
        gates["forward_finite"] = False
        gates["forward_error"] = repr(exc)[:300]

    R.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = R.CKPT_DIR / f"M2_{m1_path.stem.replace('M1R_', '').replace('M1Z_', 'z_')}_{args.mode}.pth"
    new_report = dict(report)
    new_report["recalibration"] = {
        "mode": args.mode, "n_images": len(dicts), "seed": SEED_RECAL, "refine_passes": args.refine,
        "refine_images": len(refine_dicts), "init": args.init, "n_seq_images": args.n_seq_images, "seq_rounds": args.seq_rounds, "drift_between_refine_passes": drifts, "counters": counters, "match_check": check,
        "source_m1": str(m1_path),
    }
    torch.save({"model_state_dict": student.state_dict(), "pruning_report": new_report, "source": "M0-8000",
                "stage": "M2", "created": time.strftime("%Y-%m-%d %H:%M:%S")}, out_path)

    try:
        fresh = build_model(cfg)
        sd2 = torch.load(out_path, map_location="cpu", weights_only=False)["model_state_dict"]
        reconstruct_pruned_architecture(fresh, sd2)
        fresh.load_state_dict(sd2, strict=True)
        gates["reload_ok"] = True
    except Exception as exc:                                  # noqa: BLE001
        gates["reload_ok"] = False
        gates["reload_error"] = repr(exc)[:300]
    out_path.with_suffix(".recal.json").write_text(json.dumps(new_report["recalibration"] | {"gates": gates},
                                                              indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"M2 [{report['config']}]  mode={args.mode}  parametres = {n_params:,}")
    print("apparaiement avec M0 (sortie des BN) :", json.dumps(check, indent=1))
    print("PORTES :", json.dumps(gates, indent=1))
    must = ["bn_count_53", "buffers_finite", "running_var_positive", "params_unchanged", "forward_finite", "reload_ok"]
    if args.mode == "matched":
        must.append("output_std_matches_M0")
    ok = all(gates[k] for k in must)
    print("RESULTAT :", "M2 VALIDE" if ok else "M2 NON VALIDE - ne pas utiliser, envoyer ce journal")
    print(f"checkpoint : {out_path}")
    print("Suite : python evaluate_m1_report.py --checkpoint", out_path)
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inspect"); i.add_argument("--threads", type=int, default=4); i.set_defaults(fn=cmd_inspect)
    s = sub.add_parser("selftest"); s.add_argument("--n-images", type=int, default=6)
    s.add_argument("--threads", type=int, default=4); s.set_defaults(fn=cmd_selftest)
    r = sub.add_parser("recalibrate"); r.add_argument("--m1", required=True)
    r.add_argument("--mode", choices=["matched", "plain"], default="matched")
    r.add_argument("--n-images", type=int, default=300, help="'quelques centaines d'images' (rapport)")
    r.add_argument("--refine", type=int, default=0, help="passes de raffinement (statistiques gelees)")
    r.add_argument("--n-refine-images", type=int, default=150)
    r.add_argument("--init", choices=["sequential", "global"], default="sequential",
                   help="sequential : calibration couche par couche (defaut) ; global : ancienne initialisation en une passe")
    r.add_argument("--n-seq-images", type=int, default=100, help="images par groupe en mode sequentiel")
    r.add_argument("--seq-rounds", type=int, default=3, help="tours sequentiels (tour 1 a la volee, suivants gelés)")
    r.add_argument("--n-verify", type=int, default=30)
    r.add_argument("--threads", type=int, default=4); r.set_defaults(fn=cmd_recalibrate)
    args = ap.parse_args()
    R.enable_log("m2_" + args.cmd)
    args.fn(args)


if __name__ == "__main__":
    main()