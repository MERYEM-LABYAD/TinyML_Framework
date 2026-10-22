from typing import Dict, List


def box_iou(box_a, box_b):
    """
    COCO bounding boxes:
        [x, y, width, height]
    """

    ax1 = box_a[0]
    ay1 = box_a[1]
    ax2 = ax1 + box_a[2]
    ay2 = ay1 + box_a[3]

    bx1 = box_b[0]
    by1 = box_b[1]
    bx2 = bx1 + box_b[2]
    by2 = by1 + box_b[3]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def match_detections(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5,
):
    matched_targets = set()

    tp = 0
    fp = 0

    ordered_predictions = sorted(
        predictions,
        key=lambda item: item.get("score", 0.0),
        reverse=True,
    )

    for prediction in ordered_predictions:

        best_index = None
        best_iou = 0.0

        for index, target in enumerate(targets):

            if index in matched_targets:
                continue

            if prediction["class_id"] != target["class_id"]:
                continue

            current_iou = box_iou(
                prediction["box"],
                target["box"],
            )

            if current_iou > best_iou:
                best_iou = current_iou
                best_index = index

        if (
            best_index is not None
            and best_iou >= iou_threshold
        ):
            tp += 1
            matched_targets.add(best_index)
        else:
            fp += 1

    fn = len(targets) - len(matched_targets)

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    jaccard = (
        tp / (tp + fp + fn)
        if tp + fp + fn > 0
        else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
    }


def detection_metrics(
    predictions,
    targets,
    iou_threshold=0.5,
):
    result = match_detections(
        predictions,
        targets,
        iou_threshold=iou_threshold,
    )

    result["iou_threshold"] = iou_threshold

    return result