from typing import Optional, List
import pyscme
import numpy.typing as npt
import numpy as np


def comp_str_to_indices(xyz: str) -> List[int]:
    """
    Convert a string of coordinate letters into a list of corresponding indices.

    The mapping is defined as:
        'x' -> 0, 'y' -> 1, 'z' -> 2

    Args:
        xyz (str): A string containing characters ('x', 'y', 'z').

    Returns:
        List[int]: A list of integer indices corresponding to each character in the input.
    """
    indices: List[int] = []
    letter_to_number = {"x": 0, "y": 1, "z": 2}

    for i, let in enumerate(xyz):
        indices.append(letter_to_number[let])

    return indices


def get_reference_H2O_molecule(
    rOH1: float = 0.9519607159623009,
    rOH2: float = 0.9519607159623009,
    theta: Optional[float] = 1.821207441224783,
    rHH: Optional[float] = None,
    mO: float = 16.0,
    mH: float = 1.0,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Gives the position of a water-molecule in the standard orientation with the oxygen at the origin"""

    if (not theta is None) and (not rHH is None):
        raise Exception(
            "Cannot specify theta and rHH at the same time. Set one of them to `None`"
        )

    if theta is None and rHH is None:
        raise Exception("Either theta or rHH need to be specified")

    if theta is None:
        if rHH > rOH1 + rOH2:
            raise Exception(f"rHH is too large. rHH > rOH1 + rOH2 ( = {rOH1 + rOH2})")
        # Compute theta from rHH and the law of cosines
        # ... Law of cosines: rHH**2 = rOH1**2 + rOH2**2 - 2 * rOH1 * rOH2 * cos(theta)
        theta = np.arccos((rOH1**2 + rOH2**2 - rHH**2) / (2 * rOH1 * rOH2))

    # construct a water molecule with the required rOHs and theta
    rO = np.zeros(shape=(3))
    rH1 = np.array([rOH1, 0, 0])
    rH2 = np.array([np.cos(theta) * rOH2, np.sin(theta) * rOH2, 0])

    box = pyscme.SimulationBoxInfo()
    box.pbc = [False, False, False]

    # then we find the rotation matrix from the reference frame of this molecule to the global frame
    rotation_matrix = np.array(
        pyscme.get_local_frame(rO, rH1, rH2, mO, mH, box).rotation_matrix
    )

    # applying this rotation matrix in reverse, transforms these positions into the standard reference frame
    rO = rotation_matrix.T @ rO
    rH1 = rotation_matrix.T @ rH1
    rH2 = rotation_matrix.T @ rH2

    # return the positions of the water molecule
    return (rO, rH1, rH2)


class MomentExpansion:
    def __init__(
        self, r_e, theta_e, coeffs, exponents_i, exponents_j, exponents_k, mO, mH
    ):
        self.rank = len(coeffs[0].shape)
        self.r_e = r_e
        self.theta_e = theta_e
        self.coeffs = coeffs
        self.exponents_i = exponents_i
        self.exponents_j = exponents_j
        self.exponents_k = exponents_k
        self.mO = mO
        self.mH = mH
        self.box = pyscme.SimulationBoxInfo()
        self.box.pbc = [False, False, False]

        if self.rank == 1:
            expansion_func = pyscme.monomer.MomentExpansion3
        elif self.rank == 2:
            expansion_func = pyscme.monomer.MomentExpansion33
        elif self.rank == 3:
            expansion_func = pyscme.monomer.MomentExpansion333
        elif self.rank == 4:
            expansion_func = pyscme.monomer.MomentExpansion3333

        self.expansion = expansion_func(
            r_e, theta_e, coeffs, exponents_i, exponents_j, exponents_k
        )

    def moment(
        self, rO: npt.NDArray, rH1: npt.NDArray, rH2: npt.NDArray
    ) -> npt.NDArray:

        internal_geometry = pyscme.get_internal_geometry_info(rO, rH1, rH2, self.box)
        local_frame = pyscme.get_local_frame(rO, rH1, rH2, self.mO, self.mH, self.box)

        moment, derivative = self.expansion.moment_and_derivatives(
            internal_geometry, local_frame
        )

        return moment

    def moment_from_bonds(self, rOH1: float, rOH2: float, theta: float) -> npt.NDArray:
        rO, rH1, rH2 = get_reference_H2O_molecule(rOH1, rOH2, theta)
        return self.moment(rO, rH1, rH2)


class MonomerEnergy:

    def __init__(
        self,
        r_e,
        theta_e,
        coeffs,
        exponents_i,
        exponents_j,
        exponents_k,
        mO,
        mH,
        alphaoh,
        deoh,
        roh,
        phh1,
        phh2,
        energy_correction,
        beta,
    ):

        self.r_e = r_e
        self.theta_e = theta_e

        self.mO = mO
        self.mH = mH

        self.expansion = pyscme.monomer.MonomerEnergy(
            r_e, theta_e, coeffs, exponents_i, exponents_j, exponents_k
        )

        self.expansion.alphaoh = alphaoh
        self.expansion.deoh = deoh
        self.expansion.roh = roh
        self.expansion.phh1 = phh1
        self.expansion.phh2 = phh2
        self.expansion.energy_correction = energy_correction
        self.expansion.beta = beta

    def energy(
        self, rO: npt.NDArray, rH1: npt.NDArray, rH2: npt.NDArray
    ) -> npt.NDArray:

        internal_geometry = pyscme.get_internal_geometry_info(rO, rH1, rH2, self.box)
        local_frame = pyscme.get_local_frame(rO, rH1, rH2, self.mO, self.mH, self.box)

        energy, derivative = self.expansion.energy_and_force(
            internal_geometry, local_frame
        )

        return energy

    def energy_from_bonds(self, rOH1: float, rOH2: float, theta: float) -> npt.NDArray:
        rO, rH1, rH2 = get_reference_H2O_molecule(rOH1, rOH2, theta)
        return self.energy(rO, rH1, rH2)
