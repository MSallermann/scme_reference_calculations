from snakemake.script import snakemake
import pandas as pd
from pathlib import Path
import numpy as np


def main(
    base_path: Path,
    prefix: str,
    samples: list[int],
    output_path: Path,
):
    output_path = Path(output_path)
    data = dict(
        sample=[],
        energy=[],
        dipole=[],
        quadrupole=[],
    )

    for s in samples:
        xyz = base_path / f"{prefix}{s}.xyz"
        energy = np.load(base_path / f"{prefix}{s}-E.npy").tolist()
        dipole = np.load(base_path / f"{prefix}{s}-D.npy").tolist()
        quadrupole = np.load(base_path / f"{prefix}{s}-Q.npy").tolist()

        data["sample"].append(s)
        data["energy"].append(energy)
        data["dipole"].append(dipole)
        data["quadrupole"].append(quadrupole)

    df = pd.DataFrame(data)
    with open(output_path, "w") as f:
        if output_path.suffix == ".csv":
            df.to_csv(f)
        elif output_path.suffix == ".json":
            df.to_json(f, indent=4)
        elif output_path.suffix == ".hdf5":
            df.to_hdf(f, key="data")
        else:
            raise Exception(f"{output_path.suffix} is not a valid file extension")


if __name__ == "__main__":
    main(
        base_path=Path(snakemake.params.base_path),
        prefix=snakemake.params.prefix,
        samples=snakemake.params.samples,
        output_path=snakemake.output[0],
    )
