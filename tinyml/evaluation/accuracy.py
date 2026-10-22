def f1_score(precision, recall):
    denominator = precision + recall

    if denominator == 0:
        return 0.0

    return (
        2.0
        * precision
        * recall
        / denominator
    )