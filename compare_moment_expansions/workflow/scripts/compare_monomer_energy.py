from pathlib import Path
import h5py
import numpy as np
import numpy.typing as npt

from lib.util import MonomerEnergy, comp_str_to_indices

from ase.units import Bohr, Hartree
import json
import matplotlib.pyplot as plt


def main(
    scme_expansion_hdf5: Path,
    key: str,
    fitted_energy_data_set_hdf5: Path,
    geometry_key: str,
    results_key: str,
    output_path: Path,
):
    # =========== Construct the moment expansion ===========
    scme_expansions_hdf5_file = h5py.File(scme_expansion_hdf5, "r")
    scme_expansion = scme_expansions_hdf5_file[key]

    coefficients = np.array([[c] for c in scme_expansion["coefficients"]], dtype=float)
    exponents_i = np.array(scme_expansion["exponents_i"], dtype=int)
    exponents_j = np.array(scme_expansion["exponents_j"], dtype=int)
    exponents_k = np.array(scme_expansion["exponents_k"], dtype=int)

    r_e = np.array(scme_expansion["r_e"])
    theta_e = np.array(scme_expansion["theta_e"])

    alphaoh = np.array(scme_expansion["alphaoh"])
    deoh = np.array(scme_expansion["deoh"])
    phh1 = np.array(scme_expansion["phh1"])
    phh2 = np.array(scme_expansion["phh2"])
    energy_correction = np.array(scme_expansion["energy_correction"])
    beta = np.array(scme_expansion["beta"])

    monomer_energy = MonomerEnergy(
        r_e,
        theta_e,
        coefficients,
        exponents_i,
        exponents_j,
        exponents_k,
        mO=16.0,
        mH=1.0,
        alphaoh=alphaoh,
        deoh=deoh,
        roh=r_e,
        phh1=phh1,
        phh2=phh2,
        energy_correction=energy_correction,
        beta=beta,
    )

    # =========== Load the validation datasets ===========
    fitted_energy_data_set_hdf5 = h5py.File(fitted_energy_data_set_hdf5, "r")
    geometries = fitted_energy_data_set_hdf5[geometry_key]
    results = fitted_energy_data_set_hdf5[results_key]

    # Iterate over all geometry samples and compute the moment with the scme moment expansion
    # record the maximum difference in any moment component
    max_diff = 0
    diffs = []

    energies_anoop = []
    energies_scme = []

    for (rOH1, rOH2, theta_deg, rHH), energy in zip(geometries, results):

        print(rOH1, rOH2, theta_deg, rHH)

        # SCME uses Bohr and radians, so we transform the units before computing the moment
        rOH1_scme = rOH1 / Bohr
        rOH2_scme = rOH2 / Bohr
        theta_scme = theta_deg * np.pi / 180
        rHH_scme = rHH / Bohr

        energy_scme = monomer_energy.energy_from_bonds(rOH1_scme, rOH2_scme, theta_scme)

        print(energy)
        print(energy / Hartree)
        print(energy_scme)

        energies_scme.append(energy_scme)
        energies_anoop.append(energy / Hartree)

        diffs.append(np.abs(energy / Hartree - energy_scme))

    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)

    plt.plot(np.array(energies_scme) - np.array(energies_anoop), marker=".", label="scme", ls="None")
    plt.yscale('log')
    plt.xlabel("idx_sample")
    plt.ylabel(r"$\Delta$ E [H]")
    plt.legend()
    plt.savefig("monomer_energies.png", dpi=300)

    print(f"{max_diff =}")

    with open(output_path, "w") as f:
        res = dict(max_diff=max_diff, mean_diff=mean_diff, std_diff=std_diff)
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
        )
    else:
        scme_expansion_hdf5 = Path(
            "/home/moritz/SCME/scme_reference_calculations/compare_moment_expansions/resources/scme_expansions.hdf5"
        )
        key = "energy"

        fitted_energy_data_set_hdf5 = Path(
            "/home/moritz/SCME/scme_reference_calculations/compare_moment_expansions/resources/fitted_energies.hdf5"
        )

        geometry_key = "energy/geometries/test"
        results_key = "energy/test/pred"

        main(
            scme_expansion_hdf5,
            key,
            fitted_energy_data_set_hdf5,
            geometry_key,
            results_key,
            output_path="res.json",
        )
