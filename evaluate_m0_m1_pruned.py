from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any

import torch
from detectron2.layers import FrozenBatchNorm2d
import torch.nn as nn

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    detection_utils as utils,

)
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"E:\Stage2\TinyML_Framework")
FASTER_RCNN_ROOT = Path(r"E:\Stage2\Faster-rcnn")



TEST_DATASET_NAME = "my_dataset_test"

TEST_JSON = Path(
    r"E:\Stage2\IDID DATASET\testing_data.json"
)

TEST_IMAGES = Path(
    r"E:\Stage2\IDID DATASET\Train_IDID_V1.2\Train\Images"
)


# Supprimer une ancienne registration
if TEST_DATASET_NAME in DatasetCatalog.list():
    DatasetCatalog.remove(TEST_DATASET_NAME)

if TEST_DATASET_NAME in MetadataCatalog.list():
    MetadataCatalog.remove(TEST_DATASET_NAME)


# Registration EXACTEMENT comme dans le notebook
register_coco_instances(
    TEST_DATASET_NAME,
    {},
    str(TEST_JSON),
    str(TEST_IMAGES)
)

MetadataCatalog.get(TEST_DATASET_NAME).thing_classes = [
    "No issues",
    "Broken",
    "Flashover damage"
]

M0_CHECKPOINT = (
    FASTER_RCNN_ROOT
    / "output_5000"
    / "model_final.pth"
)

M1_10_CHECKPOINT = (
    PROJECT_ROOT
    / "campaign_results"
    / "checkpoints"
    / "M1_taylor_structured_pruned_r0.10.pth"
)

M1_20_CHECKPOINT = (
    PROJECT_ROOT
    / "campaign_results"
    / "checkpoints"
    / "M1_taylor_structured_pruned_r0.20.pth"
)

M1_30_CHECKPOINT = (
    PROJECT_ROOT
    / "campaign_results"
    / "checkpoints"
    / "M1_taylor_structured_pruned_r0.30.pth"
)

OUTPUT_DIR = (
    FASTER_RCNN_ROOT
    / "analysis_m0_5000"
    / "pruning_evaluation"
)

COCO_DIR = OUTPUT_DIR / "coco"

RESULTS_JSON = OUTPUT_DIR / "M0_M1_pruning_evaluation.json"

DATASET_NAME = "my_dataset_test"

NUM_CLASSES = 3

DEVICE = torch.device("cpu")

CONFIDENCE_THRESHOLD = 0.5

IOU_THRESHOLD = 0.5


# ============================================================
# EXPECTED PARAMETER COUNTS
# ============================================================

EXPECTED_PARAMETERS = {
    "M0": 41_305_311,
    "M1_10": 37_174_674,
    "M1_20": 33_039_767,
    "M1_30": 29_192_305,

}


# ============================================================
# UTILS
# ============================================================

def log(message: str = "") -> None:
    print(message, flush=True)


def count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def box_iou(
    box_a,
    box_b,
) -> float:
    """
    COCO format:
        [x, y, width, height]
    """

    ax1 = float(box_a[0])
    ay1 = float(box_a[1])
    ax2 = ax1 + float(box_a[2])
    ay2 = ay1 + float(box_a[3])

    bx1 = float(box_b[0])
    by1 = float(box_b[1])
    bx2 = bx1 + float(box_b[2])
    by2 = by1 + float(box_b[3])

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(
        0.0,
        inter_x2 - inter_x1,
    )

    inter_h = max(
        0.0,
        inter_y2 - inter_y1,
    )

    intersection = inter_w * inter_h

    area_a = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


def match_predictions(
    predictions,
    ground_truths,
    iou_threshold: float = 0.5,
):
    """
    Match predictions to GT.

    Same class required.
    One GT can be matched once.
    Predictions sorted by confidence.
    """

    predictions = sorted(
        predictions,
        key=lambda item: item.get("score", 0.0),
        reverse=True,
    )

    matched_gt = set()

    tp = 0
    fp = 0

    for prediction in predictions:

        best_iou = 0.0
        best_gt_index = None

        for gt_index, ground_truth in enumerate(
            ground_truths
        ):

            if gt_index in matched_gt:
                continue

            if (
                prediction["category_id"]
                != ground_truth["category_id"]
            ):
                continue

            current_iou = box_iou(
                prediction["bbox"],
                ground_truth["bbox"],
            )

            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_index = gt_index

        if (
            best_gt_index is not None
            and best_iou >= iou_threshold
        ):
            tp += 1
            matched_gt.add(best_gt_index)
        else:
            fp += 1

    fn = len(ground_truths) - len(matched_gt)

    return tp, fp, fn


# ============================================================
# DATASET
# ============================================================

def register_test_dataset():
    if DATASET_NAME in DatasetCatalog.list():
        DatasetCatalog.remove(DATASET_NAME)

    if DATASET_NAME in MetadataCatalog.list():
        MetadataCatalog.remove(DATASET_NAME)

    register_coco_instances(
        DATASET_NAME,
        {},
        str(TEST_JSON),
        str(TEST_IMAGES),
    )

    MetadataCatalog.get(DATASET_NAME).thing_classes = [
        "No issues",
        "Broken",
        "Flashover damage",
    ]

def get_test_dataset_dicts():
    return DatasetCatalog.get(DATASET_NAME)


# ============================================================
# CONFIG
# ============================================================

def build_cfg():
    cfg = get_cfg()

    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.MODEL.WEIGHTS = str(M0_CHECKPOINT)

    cfg.DATASETS.TEST = (
        DATASET_NAME,
    )

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        CONFIDENCE_THRESHOLD
    )

    cfg.MODEL.DEVICE = "cpu"

    return cfg


# ============================================================
# BASE MODEL
# ============================================================

def build_base_model(cfg):

    model = build_model(cfg)

    model.to(DEVICE)

    return model


# ============================================================
# PRUNED MODULE RECONSTRUCTION
# ============================================================

def replace_conv_out_channels(
    parent: nn.Module,
    attribute: str,
    checkpoint_weight: torch.Tensor,
    checkpoint_bias: torch.Tensor | None,
):
    old_module = getattr(
        parent,
        attribute,
    )

    if not isinstance(
        old_module,
        nn.Conv2d,
    ):
        raise TypeError(
            f"{attribute} is not Conv2d: "
            f"{type(old_module).__name__}"
        )

    new_out_channels = checkpoint_weight.shape[0]

    new_module = nn.Conv2d(
        in_channels=old_module.in_channels,
        out_channels=new_out_channels,
        kernel_size=old_module.kernel_size,
        stride=old_module.stride,
        padding=old_module.padding,
        dilation=old_module.dilation,
        groups=old_module.groups,
        bias=old_module.bias is not None,
        padding_mode=old_module.padding_mode,
    )

    new_module = new_module.to(
        device=checkpoint_weight.device,
        dtype=checkpoint_weight.dtype,
    )

    new_module.weight.data.copy_(
        checkpoint_weight
    )

    if (
        checkpoint_bias is not None
        and new_module.bias is not None
    ):
        new_module.bias.data.copy_(
            checkpoint_bias
        )

    setattr(
        parent,
        attribute,
        new_module,
    )


def replace_linear_out_channels(
    parent: nn.Module,
    attribute: str,
    checkpoint_weight: torch.Tensor,
    checkpoint_bias: torch.Tensor | None,
):
    old_module = getattr(
        parent,
        attribute,
    )

    if not isinstance(
        old_module,
        nn.Linear,
    ):
        raise TypeError(
            f"{attribute} is not Linear: "
            f"{type(old_module).__name__}"
        )

    new_out_features = checkpoint_weight.shape[0]

    new_module = nn.Linear(
        in_features=old_module.in_features,
        out_features=new_out_features,
        bias=old_module.bias is not None,
    )

    new_module = new_module.to(
        device=checkpoint_weight.device,
        dtype=checkpoint_weight.dtype,
    )

    new_module.weight.data.copy_(
        checkpoint_weight
    )

    if (
        checkpoint_bias is not None
        and new_module.bias is not None
    ):
        new_module.bias.data.copy_(
            checkpoint_bias
        )

    setattr(
        parent,
        attribute,
        new_module,
    )


def replace_linear_in_channels(
    parent: nn.Module,
    attribute: str,
    checkpoint_weight: torch.Tensor,
    checkpoint_bias: torch.Tensor | None,
):
    old_module = getattr(
        parent,
        attribute,
    )

    if not isinstance(
        old_module,
        nn.Linear,
    ):
        raise TypeError(
            f"{attribute} is not Linear: "
            f"{type(old_module).__name__}"
        )

    new_in_features = checkpoint_weight.shape[1]

    new_module = nn.Linear(
        in_features=new_in_features,
        out_features=old_module.out_features,
        bias=old_module.bias is not None,
    )

    new_module = new_module.to(
        device=checkpoint_weight.device,
        dtype=checkpoint_weight.dtype,
    )

    new_module.weight.data.copy_(
        checkpoint_weight
    )

    if (
        checkpoint_bias is not None
        and new_module.bias is not None
    ):
        new_module.bias.data.copy_(
            checkpoint_bias
        )

    setattr(
        parent,
        attribute,
        new_module,
    )

def replace_conv_in_channels(
    parent: nn.Module,
    attribute: str,
    checkpoint_weight: torch.Tensor,
    checkpoint_bias: torch.Tensor | None,
):
    old_module = getattr(
        parent,
        attribute,
    )

    if not isinstance(
        old_module,
        nn.Conv2d,
    ):
        raise TypeError(
            f"{attribute} is not Conv2d: "
            f"{type(old_module).__name__}"
        )

    new_in_channels = checkpoint_weight.shape[1]
    new_out_channels = checkpoint_weight.shape[0]

    new_module = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=new_out_channels,
        kernel_size=old_module.kernel_size,
        stride=old_module.stride,
        padding=old_module.padding,
        dilation=old_module.dilation,
        groups=old_module.groups,
        bias=old_module.bias is not None,
        padding_mode=old_module.padding_mode,
    )

    new_module = new_module.to(
        device=checkpoint_weight.device,
        dtype=checkpoint_weight.dtype,
    )

    new_module.weight.data.copy_(
        checkpoint_weight
    )

    if (
        checkpoint_bias is not None
        and new_module.bias is not None
    ):
        new_module.bias.data.copy_(
            checkpoint_bias
        )

    setattr(
        parent,
        attribute,
        new_module,
    )


def replace_linear_in_out_channels(
    parent: nn.Module,
    attribute: str,
    checkpoint_weight: torch.Tensor,
    checkpoint_bias: torch.Tensor | None,
):
    old_module = getattr(
        parent,
        attribute,
    )

    if not isinstance(
        old_module,
        nn.Linear,
    ):
        raise TypeError(
            f"{attribute} is not Linear: "
            f"{type(old_module).__name__}"
        )

    new_out_features = checkpoint_weight.shape[0]
    new_in_features = checkpoint_weight.shape[1]

    new_module = nn.Linear(
        in_features=new_in_features,
        out_features=new_out_features,
        bias=old_module.bias is not None,
    )

    new_module = new_module.to(
        device=checkpoint_weight.device,
        dtype=checkpoint_weight.dtype,
    )

    new_module.weight.data.copy_(
        checkpoint_weight
    )

    if (
        checkpoint_bias is not None
        and new_module.bias is not None
    ):
        new_module.bias.data.copy_(
            checkpoint_bias
        )

    setattr(
        parent,
        attribute,
        new_module,
    )

def reconstruct_pruned_architecture(model, state_dict):
    """Reconstruit les dimensions du mod?le sans supprimer les sous-couches Detectron2."""
    import torch
    import torch.nn as nn
    from detectron2.layers import Conv2d as DetectronConv2d
    from detectron2.layers import FrozenBatchNorm2d

    def get_parent(root, dotted_name):
        parts = dotted_name.split(".")
        parent = root
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    replacements = []

    # 1. Reconstruire les convolutions en pr?servant norm et activation.
    for name, module in list(model.named_modules()):
        if not name or not isinstance(module, nn.Conv2d):
            continue

        key = name + ".weight"
        if key not in state_dict:
            continue

        weight = state_dict[key]
        if weight.ndim != 4:
            continue

        out_channels = int(weight.shape[0])
        in_channels = int(weight.shape[1]) * int(module.groups)
        expected_shape = tuple(weight.shape)

        if tuple(module.weight.shape) == expected_shape:
            continue

        kwargs = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
        )

        if isinstance(module, DetectronConv2d):
            new_module = DetectronConv2d(
                **kwargs,
                norm=getattr(module, "norm", None),
                activation=getattr(module, "activation", None),
            )
        else:
            new_module = nn.Conv2d(**kwargs)

        replacements.append((name, new_module))

    for name, new_module in replacements:
        parent, child = get_parent(model, name)
        setattr(parent, child, new_module)

    # 2. Adapter les couches FrozenBatchNorm aux dimensions du checkpoint.
    replacements = []
    for name, module in list(model.named_modules()):
        if not name or not isinstance(module, FrozenBatchNorm2d):
            continue

        key = name + ".weight"
        if key not in state_dict:
            continue

        target_features = int(state_dict[key].numel())
        current_features = int(module.weight.numel())

        if current_features != target_features:
            new_norm = FrozenBatchNorm2d(
                target_features,
                eps=module.eps,
            )
            replacements.append((name, new_norm))

    for name, new_module in replacements:
        parent, child = get_parent(model, name)
        setattr(parent, child, new_module)

    # 3. Adapter les couches lin?aires si leurs dimensions ont chang?.
    replacements = []
    for name, module in list(model.named_modules()):
        if not name or not isinstance(module, nn.Linear):
            continue

        key = name + ".weight"
        if key not in state_dict or state_dict[key].ndim != 2:
            continue

        weight = state_dict[key]
        out_features, in_features = map(int, weight.shape)
        bias_key = name + ".bias"
        has_bias = bias_key in state_dict

        if (
            module.in_features != in_features
            or module.out_features != out_features
            or (module.bias is not None) != has_bias
        ):
            replacements.append((
                name,
                nn.Linear(in_features, out_features, bias=has_bias),
            ))

    for name, new_module in replacements:
        parent, child = get_parent(model, name)
        setattr(parent, child, new_module)

    # 4. Refuser toute reconstruction qui ne correspond pas aux tenseurs.
    # Synchroniser la liste r?ellement utilis?e par FPN.forward().
    fpn = getattr(getattr(model, 'backbone', None), 'bottom_up', None)
    # Le FPN Detectron2 est model.backbone, pas bottom_up.
    fpn = getattr(model, 'backbone', None)
    if fpn is not None and hasattr(fpn, 'lateral_convs'):
        for index, level in enumerate((5, 4, 3, 2)):
            layer = getattr(fpn, f'fpn_lateral{level}', None)
            if layer is not None:
                fpn.lateral_convs[index] = layer
                print(
                    f'[FPN SYNC] lateral{level} : '
                    f'{layer.in_channels} -> {layer.out_channels}'
                )

    # Synchroniser les convolutions de sortie du FPN.
    fpn = getattr(model, 'backbone', None)
    if fpn is not None and hasattr(fpn, 'output_convs'):
        for index, level in enumerate((5, 4, 3, 2)):
            layer = getattr(fpn, f'fpn_output{level}', None)
            if layer is not None:
                fpn.output_convs[index] = layer
                print(
                    f'[FPN OUTPUT SYNC] output{level} : '
                    f'{layer.in_channels} -> {layer.out_channels}'
                )

    model_state = model.state_dict()
    missing = sorted(set(state_dict) - set(model_state))
    mismatched = sorted(
        (key, tuple(model_state[key].shape), tuple(value.shape))
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape != value.shape
    )

    if missing or mismatched:
        details = []
        details.extend("Cl? absente du mod?le : " + key for key in missing[:30])
        details.extend(
            f"Dimensions diff?rentes : {key} mod?le={model_shape}, checkpoint={ckpt_shape}"
            for key, model_shape, ckpt_shape in mismatched[:30]
        )
        raise RuntimeError(
            "Reconstruction encore incompl?te :\n" + "\n".join(details)
        )

    print("[OK] Reconstruction des dimensions termin?e.")
    print(f"[INFO] Tenseurs du checkpoint v?rifi?s : {len(state_dict)}")
    return model


def load_pruned_model(
    cfg,
    checkpoint_path: Path,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"'model_state_dict' absent from "
            f"{checkpoint_path}"
        )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    model = build_base_model(cfg)

    reconstruct_pruned_architecture(
        model=model,
        state_dict=state_dict,
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing:
        raise RuntimeError(
            f"Missing keys while loading "
            f"{checkpoint_path}:\n"
            + "\n".join(missing)
        )

    if unexpected:
        raise RuntimeError(
            f"Unexpected keys while loading "
            f"{checkpoint_path}:\n"
            + "\n".join(unexpected)
        )

    model.eval()

    return model, checkpoint


# ============================================================
# CUSTOM DETECTION METRICS
# ============================================================

def calculate_detection_metrics(
    model,
    cfg,
):
    tp_total = 0
    fp_total = 0
    fn_total = 0

    model.eval()

    start_time = time.perf_counter()

    test_loader = build_detection_test_loader(
        cfg,
        DATASET_NAME,
        num_workers=0,
    )

    with torch.no_grad():
        for image_index, inputs in enumerate(
            test_loader,
            start=1,
        ):
            outputs = model(inputs)

            for input_data, output in zip(
                inputs,
                outputs,
            ):
                instances = output["instances"].to("cpu")

                predictions = []

                for index in range(len(instances)):
                    box = (
                        instances.pred_boxes[index]
                        .tensor[0]
                        .tolist()
                    )

                    x1, y1, x2, y2 = box

                    predictions.append(
                        {
                            "bbox": [
                                x1,
                                y1,
                                x2 - x1,
                                y2 - y1,
                            ],
                            "category_id": int(
                                instances.pred_classes[index]
                            ),
                            "score": float(
                                instances.scores[index]
                            ),
                        }
                    )

                ground_truths = []

                for annotation in input_data.get(
                    "annotations",
                    [],
                ):
                    ground_truths.append(
                        {
                            "bbox": annotation["bbox"],
                            "category_id": int(
                                annotation["category_id"]
                            ),
                        }
                    )

                predictions = [
                    prediction
                    for prediction in predictions
                    if prediction["score"]
                    >= CONFIDENCE_THRESHOLD
                ]

                tp, fp, fn = match_predictions(
                    predictions=predictions,
                    ground_truths=ground_truths,
                    iou_threshold=IOU_THRESHOLD,
                )

                tp_total += tp
                fp_total += fp
                fn_total += fn

            if image_index % 20 == 0:
                log(
                    f"Custom metrics: "
                    f"{image_index}/"
                    f"{len(test_loader)}"
                )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    precision_denominator = (
        tp_total + fp_total
    )

    recall_denominator = (
        tp_total + fn_total
    )

    jaccard_denominator = (
        tp_total
        + fp_total
        + fn_total
    )

    precision = (
        tp_total / precision_denominator
        if precision_denominator
        else 0.0
    )

    recall = (
        tp_total / recall_denominator
        if recall_denominator
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )

    jaccard = (
        tp_total / jaccard_denominator
        if jaccard_denominator
        else 0.0
    )

    return {
        "TP": tp_total,
        "FP": fp_total,
        "FN": fn_total,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Jaccard": jaccard,
        "custom_evaluation_seconds": elapsed,
    }


# ============================================================
# COCO EVALUATION
# ============================================================

def evaluate_coco(model, cfg, model_name):

    model_cfg = cfg.clone()

    model_cfg.DATASETS.TEST = (TEST_DATASET_NAME,)
    model_cfg.MODEL.DEVICE = "cpu"
    model_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    model_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    evaluator_output = COCO_DIR / model_name
    evaluator_output.mkdir(parents=True, exist_ok=True)

    evaluator = COCOEvaluator(
        TEST_DATASET_NAME,
        output_dir=str(evaluator_output)
    )

    test_loader = build_detection_test_loader(
        model_cfg,
        TEST_DATASET_NAME
    )
    
    results = inference_on_dataset(
        model,
        test_loader,
        evaluator
    )
    

    coco = results["bbox"]

    return {
        "AP": coco["AP"],
        "AP50": coco["AP50"],
        "AP75": coco["AP75"],
    }

# ============================================================
# ONE MODEL
# ============================================================

def evaluate_model(
    model_name: str,
    model: nn.Module,
    cfg,
    dataset_dicts,
    checkpoint_metadata=None,
):

    log("")
    log("=" * 70)
    log(f"EVALUATION: {model_name}")
    log("=" * 70)

    parameters = count_parameters(
        model
    )

    trainable_parameters = (
        count_trainable_parameters(
            model
        )
    )

    expected = EXPECTED_PARAMETERS[
        model_name
    ]

    log(
        f"Parameters: {parameters:,}"
    )

    log(
        f"Expected:   {expected:,}"
    )

    if parameters != expected:

        raise RuntimeError(
            f"{model_name}: parameter count mismatch. "
            f"Expected {expected}, got {parameters}."
        )

    coco_results = evaluate_coco(
        model=model,
        cfg=cfg,
        model_name=model_name,
    )

    log(
        f"{model_name} COCO evaluation finished."
    )

    custom_results = (
        calculate_detection_metrics(
            model=model,
            cfg=cfg,

        )
    )

    log(
        f"{model_name} custom evaluation finished."
    )

    reduction = (
        100.0
        * (
            EXPECTED_PARAMETERS["M0"]
            - parameters
        )
        / EXPECTED_PARAMETERS["M0"]
    )

    result = {
        "model": model_name,
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "parameter_reduction_percent": reduction,
        "coco": coco_results,
        "custom_detection": custom_results,
    }

    if checkpoint_metadata is not None:

        result[
            "checkpoint_metadata"
        ] = {
            key: value
            for key, value
            in checkpoint_metadata.items()
            if key != "model_state_dict"
        }

    log(
        json.dumps(
            result,
            indent=4,
            default=str,
        )
    )

    return result


# -------------------------
def debug_model_predictions(model, cfg, model_name):
    from detectron2.data import DatasetCatalog
    import cv2
    import torch

    dataset = DatasetCatalog.get(DATASET_NAME)
    sample = dataset[0]

    image = cv2.imread(sample["file_name"])

    if image is None:
        raise RuntimeError(f"Impossible de lire : {sample['file_name']}")

    height, width = image.shape[:2]

    inputs = {
        "image": torch.as_tensor(
            image.astype("float32").transpose(2, 0, 1)
        ),
        "height": height,
        "width": width,
    }

    model.eval()

    with torch.no_grad():
        outputs = model([inputs])

    instances = outputs[0]["instances"].to("cpu")

    print()
    print("=" * 60)
    print(f"DEBUG PREDICTIONS: {model_name}")
    print("=" * 60)
    print("Image:", sample["file_name"])
    print("Image size:", width, "x", height)
    print("Number of predictions:", len(instances))

    if len(instances) > 0:
        print("Scores:", instances.scores[:20].tolist())
        print("Classes:", instances.pred_classes[:20].tolist())
        print("Boxes:", instances.pred_boxes.tensor[:5].tolist())
    else:
        print("NO PREDICTIONS")

# ============================================================
# MAIN
# ============================================================


def load_m0_model(cfg):
    """Construit et charge le mod?le Faster R-CNN de r?f?rence M0."""
    model = build_base_model(cfg)
    DetectionCheckpointer(model).load(str(M0_CHECKPOINT))
    model.eval()

    print(f"[M0] Checkpoint charg? : {M0_CHECKPOINT}")
    print(f"[M0] Param?tres : {count_parameters(model):,}")

    return model


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    COCO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_paths = [
        TEST_JSON,
        TEST_IMAGES,
        M0_CHECKPOINT,
        M1_10_CHECKPOINT,
        M1_20_CHECKPOINT,
        M1_30_CHECKPOINT,

    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required path:\n{path}"
            )

    register_test_dataset()

    dataset_dicts = get_test_dataset_dicts()

    log(
        f"Test images: {len(dataset_dicts)}"
    )

    cfg = build_cfg()

    all_results = {}

    # --------------------------------------------------------
    # M0
    # --------------------------------------------------------

    m0_model = load_m0_model(
        cfg
    )

    all_results["M0"] = evaluate_model(
        model_name="M0",
        model=m0_model,
        cfg=cfg,
        dataset_dicts=dataset_dicts,
    )

    del m0_model

    # --------------------------------------------------------
    # M1 10%
    # --------------------------------------------------------

    m1_10_model, m1_10_checkpoint = (
        load_pruned_model(
            cfg,
            M1_10_CHECKPOINT,
        )
    )

    all_results["M1_10"] = evaluate_model(
        model_name="M1_10",
        model=m1_10_model,
        cfg=cfg,
        dataset_dicts=dataset_dicts,
        checkpoint_metadata=m1_10_checkpoint,
    )

    del m1_10_model

    # --------------------------------------------------------
    # M1 20%
    # --------------------------------------------------------

    m1_20_model, m1_20_checkpoint = (
        load_pruned_model(
            cfg,
            M1_20_CHECKPOINT,
        )
    )

    all_results["M1_20"] = evaluate_model(
        model_name="M1_20",
        model=m1_20_model,
        cfg=cfg,
        dataset_dicts=dataset_dicts,
        checkpoint_metadata=m1_20_checkpoint,
    )

    del m1_20_model


    # --------------------------------------------------------
    # M1 30%
    # --------------------------------------------------------

    m1_30_model, m1_30_checkpoint = (
        load_pruned_model(
            cfg,
            M1_30_CHECKPOINT,
        )
    )

    all_results["M1_30"] = evaluate_model(
        model_name="M1_30",
        model=m1_30_model,
        cfg=cfg,
        dataset_dicts=dataset_dicts,
        checkpoint_metadata=m1_30_checkpoint,
    )

    del m1_30_model
    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    final_results = {
        "evaluation_protocol": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "dataset": DATASET_NAME,
            "test_json": str(TEST_JSON),
            "test_images": str(TEST_IMAGES),
        },
        "results": all_results,
    }

    with RESULTS_JSON.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            final_results,
            handle,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    log("")
    log("=" * 70)
    log("FINAL RESULTS")
    log("=" * 70)

    for model_name, result in all_results.items():

        coco = result["coco"]
        custom = result[
            "custom_detection"
        ]

        log(
            f"{model_name}: "
            f"Params={result['parameters']:,} | "
            f"Reduction="
            f"{result['parameter_reduction_percent']:.4f}% | "
            f"AP={coco['AP']} | "
            f"AP50={coco['AP50']} | "
            f"AP75={coco['AP75']} | "
            f"Precision="
            f"{custom['Precision']:.6f} | "
            f"Recall="
            f"{custom['Recall']:.6f} | "
            f"TP={custom['TP']} | "
            f"FP={custom['FP']} | "
            f"FN={custom['FN']}"
        )

    log("")
    log(
        f"Saved results to:\n{RESULTS_JSON}"
    )


if __name__ == "__main__":
    main()



