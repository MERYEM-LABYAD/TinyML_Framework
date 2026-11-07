"""Ajoute a run_m4_report.py l'option --fp32-units (ex: box_head  ou  stem,res2) : ces unites restent en float.
Idempotent. Usage : python m4_patch.py run_m4_report.py"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text(encoding="utf-8")
if "SKIP_UNITS" in s:
    print("deja patche"); sys.exit(0)
REPL = [
 ("SEED_CALIB = 1\n", "SEED_CALIB = 1\nSKIP_UNITS = set()          # unites laissees en float (option --fp32-units)\n"),
 ("    units.append((\"box_head\", model.roi_heads, \"box_head\", model.roi_heads.box_head))\n    return units\n",
  "    units.append((\"box_head\", model.roi_heads, \"box_head\", model.roi_heads.box_head))\n    return [u for u in units if u[0] not in SKIP_UNITS]\n"),
 ("    backend = pick_backend(args.backend)\n",
  "    backend = pick_backend(args.backend)\n    SKIP_UNITS.clear()\n    SKIP_UNITS.update(x for x in args.fp32_units.split(\",\") if x)\n    if SKIP_UNITS:\n        print(f\"[M4] unites laissees en FLOAT : {sorted(SKIP_UNITS)}\", flush=True)\n"),
 ("    base = m3_path.stem[3:]\n", "    base = m3_path.stem[3:] + (\"_fp32_\" + args.fp32_units.replace(\",\", \"-\") if args.fp32_units else \"\")\n"),
 ("    q.add_argument(\"--skip-eval\", action=\"store_true\")\n",
  "    q.add_argument(\"--skip-eval\", action=\"store_true\")\n    q.add_argument(\"--fp32-units\", default=\"\", help=\"unites gardees en float : stem,res2,res3,res4,res5,box_head\")\n"),
]
for old, new in REPL:
    if s.count(old) != 1:
        sys.exit(f"REPERE INTROUVABLE OU NON UNIQUE ({s.count(old)}) : {old[:70]!r} -> rien ecrit")
    s = s.replace(old, new)
p.write_text(s, encoding="utf-8"); print("OK : run_m4_report.py patche")
