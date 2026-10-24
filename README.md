<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0B7A66,100:0B1B2B&height=190&section=header&text=TinyML%20Framework&fontColor=ffffff&fontSize=52&fontAlignY=36&animation=twinkling&desc=Compressing%20a%20Faster%20R-CNN%20insulator%20detector%20for%20UAV%20inspection&descSize=17&descAlignY=60" alt="TinyML Framework" />
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&pause=1200&color=2FE3B0&center=true&vCenter=true&width=760&lines=Prune+%E2%86%92+Recalibrate+%E2%86%92+Distill+%E2%86%92+Quantize;165.6+MB+%E2%86%92+18.2+MB%2C+9.1%C3%97+smaller;Edge+AI+for+power-line+inspection+drones;Every+number+in+this+repo+is+measured%2C+and+its+limits+are+declared" alt="Typing animation" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Detectron2-0467DF?style=for-the-badge&logo=meta&logoColor=white" alt="Detectron2" />
  <img src="https://img.shields.io/badge/Edge%20AI-TinyML-0B7A66?style=for-the-badge" alt="TinyML" />
  <img src="https://img.shields.io/badge/Dataset-IDID%20V1.2-555?style=for-the-badge" alt="IDID V1.2" />
</p>


<p align="center">
  <img src="assets/hero.svg" alt="From a heavy detector to a small model on a drone" width="900" />
</p>

---

## At a glance

<table align="center">
  <tr>
    <td align="center" width="25%"><h2>101.1 %</h2><sub>of the original AP kept after removing <b>41.8 %</b> of the parameters<br/>(95.6 % against the detector retrained with the same recipe)</sub></td>
    <td align="center" width="25%"><h2>9.1×</h2><sub>smaller: <b>165.6 MB → 18.2 MB</b><br/>for the maximal model in INT8</sub></td>
    <td align="center" width="25%"><h2>80.2 %</h2><sub>of the original accuracy kept by that<br/>maximal INT8 model</sub></td>
    <td align="center" width="25%"><h2>0</h2><sub>measurements on embedded hardware so far:<br/>that is the next step</sub></td>
  </tr>
</table>

---

## Why this exists

<table>
  <tr>
    <td width="33%" valign="top"><h4>🧗 Before</h4>A technician climbs the pylon to look at each insulator. Slow, costly, and dangerous.</td>
    <td width="33%" valign="top"><h4>☁️ Drone + cloud</h4>A drone films the line and sends images to a server. Safer, but nothing works without a network.</td>
    <td width="33%" valign="top"><h4>🧠 Edge AI (this project)</h4>The detector runs <b>on the drone</b>. For that it must be small, so this framework compresses it while measuring what it costs.</td>
  </tr>
</table>

The detector finds insulators on UAV images and tells whether each is **healthy** (*No issues*), **broken** (*Broken*) or damaged by an electric arc (*Flashover damage*). The original network (Faster R-CNN, ResNet-50-FPN, 41.3 M parameters, AP 71.19 on 320 test images) is too heavy for a small onboard computer.

This repository belongs to an internship carried out at **Sport Score** (EMG Rabat, TinyML / Edge AI). Models are compared by **retention**: the AP of the compressed model divided by the AP of the original detector.

---

## The pipeline

```mermaid
flowchart LR
    A[M0<br/>original detector<br/>8,000 it.] --> B[M1<br/>structured pruning<br/>Taylor importance]
    B --> C[M2<br/>recalibration<br/>BatchNorm statistics]
    C --> D[M3<br/>distillation<br/>detection loss + 0.5 x MSE on FPN P2-P5]
    D --> E[M4<br/>INT8 quantization<br/>stem kept in FP32]
    E --> F[(Onboard<br/>perspective)]
    style A fill:#1B5E9B,color:#fff
    style D fill:#0B7A66,color:#fff
    style E fill:#0B7A66,color:#fff
    style F fill:#555,color:#fff,stroke-dasharray: 5 5
```

| Step | What it does | Plain words |
|---|---|---|
| **M1 pruning** | Removes the channels with the lowest Taylor score, `I_c = \|W_c ∂L/∂W_c\|`, with a global rank across layers | Trim the branches that matter least. The network gets smaller but breaks (0.7 % retention at 41.8 %). |
| **M2 recalibration** | Recomputes BatchNorm statistics on images | Re-adjust the internal settings. Helps a little when pruning is mild. |
| **M3 distillation** | Retrains the pruned student to imitate the teacher on FPN levels P2 to P5, plus the detection loss | The big network teaches the small one. This is where almost everything is repaired. |
| **M4 INT8** | Post-training quantization, first layer (stem) kept in FP32 | Write the numbers on 8 bits instead of 32. Quantizing the stem alone costs 20.6 points, so it stays in FP32. |

> **The recovery recipe.** The *final recipe* of M3 is 1,600 iterations, training at 640 to 800 px, all layers trainable. The earlier *base recipe* was 800 iterations, 480 to 576 px, stem and `res2` frozen. Same aggressive student: **69 % → 85.4 % retention**, so a retention figure only means something together with its recipe.

---

## Results

*320 test images of IDID V1.2, one run unless stated.*

| Model | Parameters removed | AP (0 to 1) | Retention |
|---|---:|---:|---:|
| Original detector M0 | 0 % | 0.7119 | 100 % |
| M0 retrained with the same recipe (control) | 0 % | 0.7528 | 105.7 % |
| Whole backbone, critical modules protected | 41.8 % | 0.7199 | **101.1 %** |
| Aggressive (mean of 3 seeds) | 57.6 % | 0.6101 | 85.7 % |
| Uniform allocation, same reduction | 57.6 % | 0.7032 | **98.8 %** |
| Maximal | 61.7 % | 0.5751 | 80.8 % |
| Maximal, INT8 (stem in FP32), **18.2 MB** | 61.7 % | 0.5712 | 80.2 % |

<details>
<summary><b>Why more than 100 %?</b></summary>

The recovery stage is extra training that the original detector never received. To check it, the unpruned detector was retrained with the same recipe: it reaches AP 0.7528 (105.7 %). Against that control, the 41.8 % model keeps **95.6 %**, which is the honest number.
</details>

<details>
<summary><b>A result that went against the hypothesis</b></summary>

At the same 57.6 % reduction and the same recipe, the **uniform** allocation (98.8 %) beat the allocation that protects the stem, RPN and heads (85.4 %) and the head-protected one (82.9 %). A likely reason is that the RoI head holds many parameters for only three classes and looks redundant, but no separate experiment isolates it and each value comes from a single run.
</details>

### What compression costs

The average hides an inequality: healthy insulators stay well detected, damage classes do not.

<p align="center">
  <img src="assets/recall.svg" alt="Recall per class, original versus compressed" width="760" />
</p>

Of the 295 objects the original finds and the maximal model misses, **206** are class confusions (a damaged insulator labelled healthy), **48** are poorly localised and **41** are missed. Failure images are in the report (section *Failure Images*).

### A second architecture

The same framework, on ResNet-18 (Imagenette): 81 % of the parameters removed, 95.0 % → 92.3 % accuracy after INT8, **20.5× smaller** (44.8 MB → 2.19 MB). Taylor was the best of four importance criteria between 50 % and 80 % reduction on this model.

---

## Quick start

```bash
git clone https://github.com/MERYEM-LABYAD/TinyML_Framework.git
cd TinyML_Framework
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
```

Python 3.11+ and a CUDA GPU are recommended for training steps (CPU is enough for evaluation and quantization).

<details>
<summary><b>Reproduce the detector pipeline</b></summary>

```bash
python verify_m0_8000.py                       # gate: M0 must give AP 71.19
python run_m1_report.py prune --config expose_whole_backbone --target-reduction 41.8 --clip rank
python run_m2_report.py recalibrate --m1 m1_report/checkpoints/M1R_expose_whole_backbone_t41.8_crank.pth
python run_m3_report.py distill --m2 m1_report/checkpoints/M2_expose_whole_backbone_t41.8_crank_matched.pth \
       --iters 1600 --threads 4 --min-size 640,704,768,800 --max-size 1333 --freeze-at 0 --tag v5pure
python run_m4_report.py quantize --m3 m1_report/checkpoints/M3_maximal_t61.7_crank_matched_v5pure.pth \
       --threads 4 --fp32-units stem
python evaluate_m1_report.py --checkpoint <path to a .pth>
```

The tag `v5pure` is the final recipe. Pruning configurations: `expose_residual` (18.4 %), `expose_internal` (31.5 %), `expose_whole_backbone` (41.8 %), `aggressive` (57.6 %), `maximal` (61.7 %), `protect_head`, `uniform_local`.
</details>

<details>
<summary><b>Reproduce the ResNet-18 study</b></summary>

```bash
python run_resnet18_framework.py prepare --dataset imagenette --out r18_report --img-size 128
python run_resnet18_framework.py pipeline --engine framework --pipe-epochs 15 --dataset imagenette --out r18_report --img-size 128
```
</details>

<details>
<summary><b>Use the framework on another model</b></summary>

```python
from tinyml.core.framework import TinyMLFramework
```

Register the model through an adapter, rank channels with an importance class and the global ranker, build a `PruningPolicy`, and prune with `StructuredPruner`. `run_resnet18_framework.py` is a complete working example (`make_framework`, `fw_register`, `prune_model_fw`, `fw_validate`).
</details>

---

## Data and weights (not stored in Git)

GitHub refuses files above 100 MB, and the original detector is 165.6 MB, so weights live outside the repository and are checked by hash.

| Item | Where | Check |
|---|---|---|
| IDID V1.2 images and annotations | Dataset owner (follow its licence). Set `TINYML_JSON_DIR` and `TINYML_IMAGES`. | n/a |
| Student checkpoints (about 70 MB each) | Not tracked | Hashes in [`docs/CHECKPOINTS_SHA256.csv`](docs/CHECKPOINTS_SHA256.csv) |

Environment variables: `TINYML_ROOT`, `TINYML_M0`, `TINYML_JSON_DIR`, `TINYML_IMAGES`, `TINYML_FASTER`, `TINYML_DEVICE` (`cpu` to force the CPU).

---

## Repository layout

```
TinyML_Framework/
├── tinyml/                     the framework
│   ├── adapters/               Detectron2, torchvision, plain PyTorch
│   ├── core/                   entry point, model analyzer, registry
│   ├── dependency/             dependency graph and channel groups
│   ├── importance/             Taylor, magnitude, activation, hybrid, global ranker
│   ├── pruning/                policy, protection, structured pruner, validator
│   └── distillation/ evaluation/ quantization/ recovery/ visualization/
├── run_m1_report.py … run_m4_report.py   the four steps
├── verify_m0_8000.py           gate on the original detector
├── evaluate_m1_report.py       evaluation of any checkpoint
├── measure_costs.py            parameters, size, GFLOPs, latency
├── run_resnet18_framework.py   ResNet-18 through the framework
├── run_resnet18_study.py       ResNet-18 with Torch-Pruning called directly (control)
├── benchmark_on_device.py      latency, memory, temperature on a target device
├── m1_report/                  detector results: logs, tables, figures
├── r18_report/                 ResNet-18 results
├── docs/                       file index, checkpoint hashes, methodology notes
└── tests/
```

---

## Next: onboard hardware

<p align="center"><img src="https://img.shields.io/badge/Raspberry%20Pi%205-C51A4A?style=for-the-badge&logo=raspberry-pi&logoColor=white" alt="Raspberry Pi 5" /> <img src="https://img.shields.io/badge/NVIDIA%20Jetson-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="NVIDIA Jetson" /></p>

The INT8 model is 18.2 MB, so memory is not the issue on a Raspberry Pi 5 (4-core Arm, 8 GB, no CUDA GPU). Speed is the open question: the best figure so far is about 5 seconds per image on a laptop CPU, which is **not real time**. The report (appendix *Embedded Platforms and Integration Path*) describes the steps: quantize with the `qnnpack` backend, export to TorchScript or ONNX, check that AP is unchanged, then measure latency, memory, temperature and energy. `benchmark_on_device.py` is ready:

```bash
python benchmark_on_device.py --model student_int8.ts --input 1 3 800 1200 --threads 4 --runs 30
