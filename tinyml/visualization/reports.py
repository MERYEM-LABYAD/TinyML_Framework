import json


def save_json_report(
    results,
    path,
):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
        )