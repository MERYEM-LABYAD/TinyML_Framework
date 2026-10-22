def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def count_trainable_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def parameter_reduction(
    baseline_parameters,
    current_parameters,
):
    if baseline_parameters <= 0:
        raise ValueError(
            "baseline_parameters must be > 0."
        )

    return (
        1.0
        - current_parameters / baseline_parameters
    ) * 100.0