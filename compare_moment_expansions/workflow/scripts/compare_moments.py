from pathlib import Path
import h5py
import numpy as np
import numpy.typing as npt
from typing import Optional
from lib.util import MomentExpansion, comp_str_to_indices

from ase.units import Bohr
import json


def main(
    scme_expansion_hdf5: Path,
    key: str,
    fitted_energy_data_set_hdf5: Path,
    geometry_key: str,
    results_key: str,
    output_path: Path,
    functional: Optional[str] = None,
    moment_str: Optional[str] = None,
):
    # =========== Construct the moment expansion ===========
    scme_expansions_hdf5_file = h5py.File(scme_expansion_hdf5, "r")
    scme_expansion = scme_expansions_hdf5_file[key]

    coefficients = np.array(scme_expansion["coefficients"], dtype=float)
    exponents_i = np.array(scme_expansion["exponents_i"], dtype=int)
    exponents_j = np.array(scme_expansion["exponents_j"], dtype=int)
    exponents_k = np.array(scme_expansion["exponents_k"], dtype=int)

    # SCME uses Bohr and radians
    r_e = np.array(scme_expansion["r_e"])
    theta_e = np.array(scme_expansion["theta_e"])

    expansion = MomentExpansion(
        r_e, theta_e, coefficients, exponents_i, exponents_j, exponents_k, mO=16, mH=1
    )

    # =========== Load the validation datasets ===========
    fitted_energy_data_set_hdf5 = h5py.File(fitted_energy_data_set_hdf5, "r")
    geometries = fitted_energy_data_set_hdf5[geometry_key]
    results = fitted_energy_data_set_hdf5[results_key]

    # Figure out which components we have to compare
    non_zero_comps = results.attrs["comp"]
    indices = [comp_str_to_indices(c) for c in non_zero_comps]

    # Iterate over all geometry samples and compute the moment with the scme moment expansion
    # record the maximum difference in any moment component
    max_diff = 0
    diffs = []

    moments_scme = []

    for (rOH1, rOH2, theta_deg), moment in zip(geometries, results):
        # SCME uses Bohr and radians, so we transform the units before computing the moment
        rOH1_scme = rOH1 / Bohr
        rOH2_scme = rOH2 / Bohr
        theta_scme = theta_deg * np.pi / 180

        moment_scme = expansion.moment_from_bonds(rOH1_scme, rOH2_scme, theta_scme)

        for i, m_component in enumerate(moment):
            m_component_scme = moment_scme[indices[i]]
            moments_scme.append(moment_scme)
            diffs.append(np.abs(m_component - m_component_scme))

    avg_moment_scme = np.mean(moments_scme, axis=0).tolist()
    avg_moment_reference = np.mean(results, axis=0).tolist()

    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)

    with open(output_path, "w") as f:
        res = dict(
            max_diff=max_diff,
            mean_diff=mean_diff,
            std_diff=std_diff,
            functional=functional,
            moment=moment_str,
            avg_moment_scme=avg_moment_scme,
            avg_moment_reference=avg_moment_reference,
        )
        json.dump(res, f, indent=4)


if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import snakemake

        main(
            scme_expansion_hdf5=snakemake.input["scme_expansions_file"],
            key=snakemake.params["key"],
            fitted_energy_data_set_hdf5=snakemake.input["fitted_energies_file"],
            geometry_key=snakemake.params["geometry_key"],
            results_key=snakemake.params["results_key"],
            output_path=snakemake.output[0],
            functional=snakemake.params["functional"],
            moment_str=snakemake.params["moment"]
        )
    else:
        scme_expansion_hdf5 = Path(
            "/home/moritz/SCME/scme_reference_calculations/compare_moment_expansions/resources/scme_expansions.hdf5"
        )
        key = "component_PBE_fullrange_reflect_4_5/dip_dip"

        fitted_energy_data_set_hdf5 = Path(
            "/home/moritz/SCME/scme_reference_calculations/compare_moment_expansions/resources/fitted_energies.hdf5"
        )

        geometry_key = "component_PBE_fullrange_reflect_4_5/geometries/test"
        results_key = "component_PBE_fullrange_reflect_4_5/dip_dip/test/pred"

        main(
            scme_expansion_hdf5,
            key,
            fitted_energy_data_set_hdf5,
            geometry_key,
            results_key,
            output_path="res.json",
        )
