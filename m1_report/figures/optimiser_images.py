r"""
Allege les images du rapport pour que Overleaf (plan gratuit) compile vite.
Usage (dans le dossier qui contient tes PNG/JPG) :
    python optimiser_images.py            -> cree le dossier images_legeres\ (originaux intacts)
    python optimiser_images.py --max 1600 -> largeur maximale en pixels (defaut 1600)
Ensuite, uploade les fichiers de images_legeres\ dans Overleaf a la place des gros.
"""
import argparse
from pathlib import Path
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--max", type=int, default=1600)
ap.add_argument("--src", default=".")
a = ap.parse_args()
src = Path(a.src)
out = src / "images_legeres"
out.mkdir(exist_ok=True)
tot0 = tot1 = 0
for p in sorted(src.iterdir()):
    if p.suffix.lower() not in (".png", ".jpg", ".jpeg") or not p.is_file():
        continue
    im = Image.open(p)
    if im.width > a.max:
        im = im.resize((a.max, round(im.height * a.max / im.width)), Image.LANCZOS)
    q = out / p.name
    if p.suffix.lower() == ".png":
        im.save(q, optimize=True)
    else:
        im.convert("RGB").save(q, quality=82, optimize=True)
    s0, s1 = p.stat().st_size / 1e6, q.stat().st_size / 1e6
    tot0 += s0; tot1 += s1
    print(f"{p.name:50s} {s0:7.2f} Mo -> {s1:6.2f} Mo")
print(f"\nTOTAL {tot0:.1f} Mo -> {tot1:.1f} Mo   (dossier : {out})")
