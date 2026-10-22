## Ordre d'exécution

```
python tests\test_report_pruner_toy.py        # 7 tests
python tests\test_frozen_bn_sync.py
python tests\test_report_graph.py             # vérifie que bbox_pred est dans le graphe
python run_m1_report.py score --n-images 300 --threads 4   # déjà fait : taylor_scores_m0_8000.pt
python run_m1_report.py calibrate              # choisit les percentiles d'écrêtage (--clip) ; log : m1_report\last_calibrate.log
python run_m1_report.py plan                   # AUCUNE chirurgie ; log : m1_report\last_plan.log
python run_m1_report.py prune --config expose_whole_backbone --target-reduction 41.8 [--clip 5,95]
```

Chaque commande écrit automatiquement son journal dans `m1_report\last_<commande>[_<config>].log` : envoyer ce fichier en cas de doute.

## Pourquoi « M1 NON VALIDE » est arrivé (corrigé)

`roi_heads.box_predictor.bbox_pred` était absent du graphe de dépendances (aucune détection sur l'image d'exemple, donc les deltas de boîtes n'étaient reliés à aucune sortie). `fc2` était coupé mais `bbox_pred` gardait 1024 entrées : erreur `1000x635 vs 1024x12`. Les portes de contrôle ont bien bloqué le checkpoint. `report_graph.py` capte les sorties de `rpn_head` et `box_predictor`. **Ne pas utiliser** `M1R_expose_whole_backbone_r0.55.pth`.

Pour chaque `prune`, le script écrit `m1_report\checkpoints\M1R_<config>_<tag>.pth` et un `.report.json` (carte des canaux retirés par bloc, portes de contrôle). **Un M1 n'est valide que si la ligne finale affiche `M1 VALIDE`.**

## Ce qui est conforme au rapport

- Score Taylor `|w · ∂L/∂w|` accumulé sur plusieurs centaines d'images, sans pas d'optimiseur (4.4.1). Défaut : 300 images.
- Couches couplées (flux résiduels) : scores **sommés** avant normalisation, même masque pour la convolution et son raccourci (4.4.2). Les couplages viennent de Torch-Pruning.
- Normalisation **par groupe** (écrêtage par percentiles puis min-max), puis `s_final = w_g · s_norm`, classement global, plancher `MIN_KEEP = 8` (4.4.3).
- Planning **one-shot** : un seul classement, une seule chirurgie (4.5).
- Poids `w_stem / w_internal / w_residual / w_rpn / w_fc` et ratios : ceux des Tables 26–38.
- Exclusions dures : bus FPN et sorties des prédicteurs (cls_score, bbox_pred, objectness, anchor_deltas). Exclusions douces par poids : stem, internes, résiduels, convolution RPN, fc1/fc2 (comme dans les expériences du rapport).

## Hypothèses à déclarer (le rapport ne les précise pas)

1. Percentiles d'écrêtage : (1, 99).
2. `FILTER_ZERO_THRESH` : avec un classement par canaux, un canal retiré est supprimé en entier (100 % de zéros), donc ce paramètre est journalisé mais sans effet. Le rapport ne décrit pas de zéros partiels pour le détecteur.
3. Sens de `GLOBAL_PRUNE_RATIO` : fraction des canaux **éligibles** retirés globalement. La réduction de paramètres est un **résultat** (comme dans le rapport). Si la prévision de `plan` s'écarte de la valeur du rapport, utiliser `--target-reduction 41.8` : le script cherche le ratio qui donne la même réduction, ce qui rend la comparaison équitable.
4. Config `uniform_local` : le mécanisme « local » du rapport n'est pas décrit ; ici tous les `w = 1`. Reproduction approximative, à ne pas mettre en avant.
5. Le rapport ne dit pas quelles images servent au calcul Taylor : ici 300 images d'entraînement, graine 0, redimensionnées à 800 px, sans retournement.

## Ce que les portes de contrôle vérifient

- aucun groupe rejeté par Torch-Pruning (sinon la raison est affichée) ;
- nombre de paramètres après chirurgie **exactement égal** à la prévision analytique ;
- passe avant sur une vraie image, valeurs finies ;
- taille du fichier cohérente avec 4 octets × paramètres (un fichier de la taille de M0 = pas d'élagage réel) ;
- rechargement du checkpoint avec la même reconstruction que l'évaluation.

## Après M1

M1 seul s'effondre dans le rapport aussi (AP ≈ 0 à 0,3). L'étape suivante est M2 (recalibration des FrozenBN), puis M3 (distillation FPN P2–P5, β = 0,5), puis M4 (INT8 partiel). `evaluate_m1_report.py` sert à contrôler qu'un checkpoint charge et prédit ; les chiffres utiles viendront après M3.


## Deuxième mécanisme : « poids mis à zéro puis filtres retirés » (tables du rapport)

**Pourquoi.** Les tables 26–38 du rapport définissent `GLOBAL_PRUNE_RATIO` comme une *fraction de poids mis à zéro globalement* et `FILTER_ZERO_THRESH` comme « un filtre est retiré si plus de X % de ses poids sont à zéro », alors que le chapitre 4 décrit un classement par canaux. Les calibrations montrent que le classement par canaux ne reproduit pas la sélectivité des figures du rapport (ex. « expose residual » : résiduels −39..−48 %, internes presque intacts ; tête fc6/fc7 à −13 %/−6 %) : avec lui, la tête fc est toujours coupée de 20–40 % et les blocs non exposés perdent 12–22 %. Le seuil de zéros amplifie un écart de poids 0,7/1,0 en un écart de coupe très net, et protège la tête (poids 3,0–3,5).

**Étapes** (`tinyml/importance/report_zeros.py`) : score Taylor *par poids* → normalisation par groupe (écrêtage + min-max) → `s_final = w_g · s_norm` → seuil global : la fraction `ratio` de poids de plus faible score est mise à zéro → un canal est retiré si la fraction de zéros de son filtre (de tous les filtres couplés pour un flux résiduel) dépasse `FILTER_ZERO_THRESH`, plancher `MIN_KEEP` → les zéros des filtres survivants sont conservés.

**Commandes** (ratios et seuils BRUTS du rapport, aucune cible de réduction imposée) :

```
python tests\test_report_zeros_toy.py
python run_m1_report.py score-weights --n-images 300 --threads 4     # ~1,5 h, reprise automatique
python run_m1_report.py zcalibrate                                      # test décisif : 7 réductions vs rapport
python run_m1_report.py prune --config expose_whole_backbone --mode zeros --group layer --clip 5,95
```

`zcalibrate` affiche, pour chaque groupe de normalisation (`layer`, `unit`, `label`, `labelstage`) et chaque écrêtage, la réduction de paramètres obtenue pour les 7 configurations face à celle du rapport (51,8 / 48,6 / 18,4 / 31,5 / 41,8 / 57,6 / 61,7 %) et les cartes de canaux par bloc. Si une combinaison reproduit les réductions SANS réglage, le mécanisme est confirmé. Les checkpoints sont nommés `M1Z_<config>_<groupe>_<écrêtage>.pth` (préfixe `M1Z`, pour ne pas les confondre avec `M1R`).
