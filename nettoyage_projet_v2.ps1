# nettoyage_projet_v2.ps1
# Range le projet selon ARBORESCENCE_FINALE_PROJET.md. Tout ce qui n'est pas dans la liste est DEPLACE
# (jamais supprime) dans _archive_ancien_essai\ . Pour annuler, remets le fichier a sa place.
# Difference avec la v1 : le framework tinyml\ est GARDE en entier (il sert a l'etude ResNet-18).
#
# Usage (depuis E:\Stage2\TinyML_Framework) :
#   powershell -ExecutionPolicy Bypass -File .\nettoyage_projet_v2.ps1 -Simulation   # montre sans rien deplacer
#   powershell -ExecutionPolicy Bypass -File .\nettoyage_projet_v2.ps1               # execute
param([switch]$Simulation)

$root = (Get-Location).Path
$arch = Join-Path $root "_archive_ancien_essai"
if (-not $Simulation) { New-Item -ItemType Directory -Force -Path $arch | Out-Null }

$keepFiles = @(
  "run_m1_report.py","run_m2_report.py","run_m3_report.py","run_m4_report.py",
  "verify_m0_8000.py","evaluate_m0_m1_pruned.py","evaluate_m1_report.py","make_m0_control.py",
  "measure_costs.py","report_visuals.py","eval_dump.py","analyze_errors.py",
  "run_resnet18_study.py","run_resnet18_framework.py",
  "requirements.txt","README.md",
  "ARBORESCENCE_FINALE_PROJET.md","nettoyage_projet_v2.ps1","nettoyage_projet.ps1"
)
# fichiers de documentation a ranger dans docs\ (ils ne sont pas archives)
$docFiles = @("README_M1_FR.md","REGISTRE_MODELES_ET_ECARTS_RAPPORT.md","GUIDE_METHODOLOGIE_RAPPORT_FR.md",
  "SUIVI_RESULTATS_M1_M3_FR.md","CLARIFICATIONS_3_POINTS_PROF_FR.md","AUDIT_REPORT.txt","SOURCE_EXTRACTS.txt",
  "FILES_INDEX.csv","CHECKPOINTS_SHA256.csv")
$keepDirs = @("tinyml","tests","m1_report","r18_report","kaggle","rapport","docs","_archive_ancien_essai")

function Move-ToArchive($item, $subdir) {
  $dest = Join-Path $arch $subdir
  if ($Simulation) { Write-Host "  [simulation] archive -> $($item.Name)"; return }
  New-Item -ItemType Directory -Force -Path $dest | Out-Null
  Move-Item -LiteralPath $item.FullName -Destination $dest -Force
  Write-Host "  archive : $($item.Name)"
}

Write-Host "=== ETAPE 1 : creation des dossiers kaggle, rapport, docs ==="
foreach ($d in @("kaggle","rapport","docs")) {
  $p = Join-Path $root $d
  if ($Simulation) { Write-Host "  [simulation] dossier $d" } else { New-Item -ItemType Directory -Force -Path $p | Out-Null }
}

Write-Host "=== ETAPE 2 : racine du projet ==="
Get-ChildItem -LiteralPath $root -Force | ForEach-Object {
  $n = $_.Name
  if ($n -eq ".pytest_cache" -or $n -eq "__pycache__") {          # caches regenerables
    if (-not $Simulation) { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
    Write-Host "  cache supprime : $n"; return
  }
  if ($n.StartsWith(".")) { return }                              # .git, .vscode ... gardes
  if ($_.PSIsContainer) {
    if ($keepDirs -notcontains $n) { Move-ToArchive $_ "racine" }
    return
  }
  if ($docFiles -contains $n) {
    if ($Simulation) { Write-Host "  [simulation] docs <- $n" }
    else { Move-Item -LiteralPath $_.FullName -Destination (Join-Path $root "docs") -Force; Write-Host "  docs : $n" }
    return
  }
  if ($n -like "kaggle_*.py" -or $n -eq "worker_policies.py") {
    if ($Simulation) { Write-Host "  [simulation] kaggle <- $n" }
    else { Move-Item -LiteralPath $_.FullName -Destination (Join-Path $root "kaggle") -Force; Write-Host "  kaggle : $n" }
    return
  }
  if ($n -like "TinyML_full_report_v12*" -or $n -like "discours_FR*") {
    if ($Simulation) { Write-Host "  [simulation] rapport <- $n" }
    else { Move-Item -LiteralPath $_.FullName -Destination (Join-Path $root "rapport") -Force; Write-Host "  rapport : $n" }
    return
  }
  if ($keepFiles -notcontains $n) { Move-ToArchive $_ "racine" }
}

Write-Host "=== ETAPE 3 : tests\ (on garde les tests du pipeline final) ==="
$keepTests = @("__init__.py","test_report_graph.py","test_report_pruner_toy.py","test_report_zeros_toy.py")
$tdir = Join-Path $root "tests"
if (Test-Path $tdir) {
  Get-ChildItem -LiteralPath $tdir -Force | ForEach-Object {
    if ($_.Name -eq "__pycache__") { if (-not $Simulation) { Remove-Item -LiteralPath $_.FullName -Recurse -Force }; return }
    if ($keepTests -notcontains $_.Name) { Move-ToArchive $_ "tests" }
  }
}

Write-Host "=== ETAPE 4 : tinyml\ garde en entier (framework), seules les sauvegardes .bak sont archivees ==="
$tin = Join-Path $root "tinyml"
if (Test-Path $tin) {
  Get-ChildItem -LiteralPath $tin -Recurse -Force -File | Where-Object { $_.Name -like "*.bak*" } | ForEach-Object { Move-ToArchive $_ "tinyml_bak" }
  if (-not $Simulation) { Get-ChildItem -LiteralPath $tin -Recurse -Directory -Filter "__pycache__" -Force | Remove-Item -Recurse -Force }
}

Write-Host ""
Write-Host "Termine. Archive : $arch"
Write-Host "Verifie : python -c `"import run_m1_report, run_resnet18_framework`"   puis relance une evaluation rapide."
Write-Host "Tes checkpoints, logs et resultats de m1_report\ n'ont pas ete touches."
