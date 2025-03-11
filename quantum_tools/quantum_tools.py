import random
import numpy as np
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile

from typing import Union
import itertools
from functools import reduce
from operator import concat

def get_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    """Return the real part of the eigenvalues of a matrix , i.e the eigenenergies of the Hamiltonian from is matrix representation
    Args:
        matrix (np.ndarray): a square matrix that represant an Hamiltonian 

    Returns:
       np.ndarray: the eigenenergies of the Hamiltonian
    """
    eigval_list = np.linalg.eig(matrix)[0]
    eigval_list[np.abs(eigval_list) < 1.e-11] = 0
    return eigval_list.real


def get_random_Observable(numQbits: int) -> str:
    """Generate a random Pauli observable of size qubit

    Args:
        numQbits (int): number of qubit

    Returns:
        str: a random Pauli observable of size num qubits
    """
    pauli = ['I', 'X', 'Y', 'Z']
    Observable = ''
    for n in range(numQbits):
        p = random.randint(0, 3)
        Observable += pauli[p]
    return Observable


def get_obs_1_qubit(numQbits: int, pos_qubit: int, obs: str):
    Observable = 'I'*numQbits
    Observable[-(1+pos_qubit)] = obs
    return Observable


def Generate_Unitary_Hermitian_Matrix(numQbits, eigenvalues):
    """
    Generate a Hermitian matrix with specified eigenvalues.

    Args:
        eigenvalues (list or np.ndarray): A list of desired eigenvalues.

    Returns:
        np.ndarray: A Hermitian matrix with the given eigenvalues.
    """
    diagonal_matrix = np.identity(2**numQbits)
    k = 0
    for eigenvalue, multiplicity in eigenvalues:
        for i in range(multiplicity):
            diagonal_matrix[k+i][k+i] = eigenvalue
        k += multiplicity
    # Generate a random unitary matrix (P)
    random_matrix = np.random.randn(
        2**numQbits, 2**numQbits) + 1j * np.random.randn(2**numQbits, 2**numQbits)
    # QR decomposition to get a unitary matrix
    Q, _ = np.linalg.qr(random_matrix)
    # Construct the Hermitian matrix: H = P \Lambda P^†
    hermitian_matrix = Q @ diagonal_matrix @ Q.conj().T
    return hermitian_matrix






def get_energy_gap(Energies: list[float], rnd: int = 4) -> list[float]:
    """Calculate the energy gap betweend different energy level.
    Remove double energy level.

    Args:
        Energies (list[float]): list of energies to calculate energy gap.

    Returns:
        list[float]: energy gap
    """
    Energies[np.abs(Energies) < 1.e-11] = 0
    Energies_no_double = []
    for energie in Energies:
        if np.round(energie, rnd) not in np.round(Energies_no_double, rnd):
            Energies_no_double.append(energie)
    res = []
    for i in range(len(Energies_no_double)-1):
        for k in range(i+1, len(Energies_no_double)):
            res.append(Energies_no_double[i]-Energies_no_double[k])
    res = np.abs(res)
    res_no_double = []
    for gap in res:
        gap = np.round(gap, rnd)
        if gap not in res_no_double:
            res_no_double.append(gap.tolist())

    res_no_double = np.array(res_no_double)
    res_no_double[np.abs(res_no_double) < 1.e-11] = 0
    return np.sort(res_no_double)


def get_multiplicity(list: list, rnd: int = 4) -> list[tuple[float, int]]:
    """Get the multiplicity of each energy level in a list and return a list of tuple with the value and his multiplicity
    """
    if isinstance(list, np.ndarray):
        list = list.tolist()
    res = []
    val = []
    for energie in list:
        if np.round(energie, rnd) not in val:
            res.append((energie, list.count(energie)))
            val.append(np.round(energie, rnd))
    return res


def get_ground_state(matrix: np.ndarray, nqubits: int, Comput_basis: bool = False):
    """return the ground_state of an Hamiltonian, i.e the state of the lowest energy from is matrix representation

    Args:
        matrix (np.ndarray): The matrix representation of the Hamiltonian 
        nqubits (int): number of qubits 
        Comput_basis (bool, optional): If True return the vector ground state as a list of tuple (coef: float, "ket in the compuational basis": float). Defaults to False.

    Returns:
        if Comput_basis is False return np.ndarray vector representing the ground state in the computational basis
        if  Comput_basis is True return a list of tuple with the coefficient and is associated ket in the compuational basis 

    """
    from itertools import product
    combinaisons = product("01", repeat=nqubits)
    computational_basis = ["".join(bits) for bits in combinaisons]
    res = np.linalg.eig(matrix)
    gr_eigenvalues = res[0].real
    gr_eigenvalues[np.abs(gr_eigenvalues) < 1.e-11] = 0
    gr_eigenvalues = np.round(gr_eigenvalues, 4)
    minimum = np.min(gr_eigenvalues)
    indices = np.where(gr_eigenvalues == minimum)[0]
    ground_state_vector = np.zeros(2**nqubits, dtype='complex128')
    for id in indices:
        ground_state_vector += res[1][:, id]

    ground_state_vector[np.abs(ground_state_vector) < 1.e-11] = 0
    ground_state_vector /= (np.linalg.vector_norm(ground_state_vector))
    if Comput_basis:
        ground_state = []
        for i in range(len(ground_state_vector)):
            ground_state.append(
                (ground_state_vector[i], computational_basis[i]))

        return ground_state
    else:
        return ground_state_vector


def get_q_local_recursive(nq, K: int) -> list[str]:
    """Generate the sequence of all the observable from 1-Pauli observable to K-Pauli observable

    Args:
        K (int): K-pauli observable to generate
    Returns:
        list[str]: list of all the observable from 1-Pauli observable to K-Pauli observable
    """
    q_local = []
    for k in range(K):
        q_local.append(get_q_local_Pauli(nq, k+1))
    return reduce(concat, q_local)


def get_q_local_Pauli(nq: int, k: int) -> list[str]:
    """Generate the sequence of all the k-Pauli observable

    Args:
        nq(int): number of qubit
        k (int):  K-pauli observable to generate

    Returns:
        list[str]:  list of all the k-Pauli observable
    """

    pauli_operators = ["X", "Y", "Z",]
    q_local = []

    all_combinations = list(itertools.product(pauli_operators, repeat=k))
    for positions in itertools.combinations(range(nq), k):
        for combination in all_combinations:
            observable = ['I'] * nq

            for i, pos in enumerate(positions):
                observable[pos] = combination[i]

            q_local.append(tuple(observable))

    return q_local


def resample_points(x_list, y_list, num_points_between=100, method='linear'):
    from scipy import interpolate
    """
    Resample coordinate data by adding points between existing points.
    
    Parameters:
    x_list (list or array): X coordinates of original points
    y_list (list or array): Y coordinates of original points
    num_points_between (int): Number of new points to add between each pair of existing points
    method (str): Interpolation method: 'linear', 'cubic', 'quadratic', etc.
                  For full options, see scipy.interpolate.interp1d documentation
    
    Returns:
    x_new (numpy array): Resampled X coordinates
    y_new (numpy array): Resampled Y coordinates
    """
    x = np.array(x_list)
    y = np.array(y_list)
    if len(x) != len(y):
        raise ValueError("x_list and y_list must have the same length")
    f = interpolate.interp1d(x, y, kind=method, assume_sorted=False)
    x_new = []
    for i in range(len(x) - 1):
        x_new.append(x[i])

        for j in range(1, num_points_between + 1):
            ratio = j / (num_points_between + 1)
            new_x = x[i] + ratio * (x[i+1] - x[i])
            x_new.append(new_x)

    x_new.append(x[-1])

    x_new = np.array(x_new)
    y_new = f(x_new)

    return x_new, y_new
