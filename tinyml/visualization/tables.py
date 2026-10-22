def stage_table(results):
    rows = []

    for stage, result in results.items():
        metrics = result["metrics"]

        rows.append(
            {
                "stage": stage,
                **metrics,
            }
        )

    return rows


def print_stage_table(results):
    rows = stage_table(results)

    if not rows:
        print("No results available.")
        return

    keys = list(rows[0].keys())

    print(
        " | ".join(keys)
    )

    print(
        "-+-".join(
            "-" * len(key)
            for key in keys
        )
    )

    for row in rows:
        print(
            " | ".join(
                str(row.get(key, ""))
                for key in keys
            )
        )