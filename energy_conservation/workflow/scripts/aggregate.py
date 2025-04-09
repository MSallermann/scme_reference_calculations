from snakemake.script import snakemake
import pandas as pd
from pathlib import Path
import json
from typing import Optional


def main(
    input_paths: list[Path],
    output_path: Path,
    added_columns: Optional[dict] = None,
    ignore_keys: Optional[list[str]] = None,
):

    data = dict()

    for idx, ip in enumerate(input_paths):
        with open(ip, "r") as f:
            res = json.load(f)

        res["file"] = str(ip)

        if not added_columns is None:
            for k in added_columns.keys():
                item = added_columns[k][idx]
                if k in data:
                    data[k].append(item)
                else:
                    data[k] = [item]

        for k, v in res.items():
            if not ignore_keys is None and k in ignore_keys:
                continue
            if k in data:
                data[k].append(v)
            else:
                data[k] = [v]

    df = pd.DataFrame(data)
    with open(output_path, "w") as f:
        df.to_csv(f)


if __name__ == "__main__":
    main(
        snakemake.input,
        snakemake.output[0],
        snakemake.params.get("added_fields", None),
        snakemake.params.get("ignore_fields", None),
    )
