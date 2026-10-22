def plot_parameter_reduction(
    stages,
    parameters,
):
    import matplotlib.pyplot as plt

    if len(stages) != len(parameters):
        raise ValueError(
            "stages and parameters must have the same length."
        )

    plt.figure()

    plt.plot(
        stages,
        parameters,
        marker="o",
    )

    plt.xlabel("Stage")
    plt.ylabel("Parameters")
    plt.title(
        "Parameter count across TinyML stages"
    )

    plt.tight_layout()
    plt.show()