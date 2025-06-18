"""
Quantum Math Utilities Module
=============================

This module provides a comprehensive set of mathematical tools tailored for quantum computing, particularly useful for quantum simulation, quantum state analysis, and Hamiltonian modeling. It includes functionality for eigenvalue computation, observable generation, ground state extraction, and mathematical filtering functions commonly used in quantum algorithm development and post-processing.

Main Capabilities:
------------------
- **Hamiltonian Analysis**:
  - Compute eigenenergies and energy gaps of quantum Hamiltonians.
  - Generate Hermitian matrices from prescribed eigenvalues.
  - Determine the ground state vector or its expansion in the computational basis.
  - Evaluate multiplicities of energy levels.

- **Pauli Observable Utilities**:
  - Generate random or structured Pauli observables.
  - Generate all $k$-local Pauli observables up to a given weight.
  - Create observables acting on a specific qubit.

- **State Vector Utilities**:
  - Sum sparse state vectors in the computational basis.
  - Convert dense state vectors to sparse dictionaries above a threshold.
  - Extract normalized ground state vectors.

- **Numerical Tools and Filters**:
  - Smooth data via interpolation and resampling.
  - Apply common spectral filters (Gaussian, Lorentzian, asymmetric decay).
  - Compute numerical derivatives.
  - Eliminate near-duplicate values from lists.

- **Utility Functions**:
  - Split lists into nearly equal-sized sublists.
  - Find the closest value in a list to a given target.

Intended Audience:
------------------
This module is designed for researchers, students, and developers working in quantum computing, quantum chemistry, or quantum algorithm design. It supports theoretical simulations, Hamiltonian diagnostics, and the generation of quantum observables in a flexible and extensible way.

Dependencies:
-------------
- **NumPy**: Core numerical and linear algebra routines.
- **SciPy** (optional): Used for smooth interpolation and resampling.
- **itertools**: For efficient combinatorial generation of basis states and observables.

Author:
-------
Hugo PAGES  
hugo.pages@etu.unistra.fr  
Date: 12/02/2025

Note:
-------
Part of the code have been writen by ChatGPT-4.

"""


import random
import numpy as np
from functools import reduce
from operator import concat
import itertools


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


def Gaussian(x, x_max=0, sigma=1):
    return np.exp(-((x - x_max) ** 2) / sigma**2)

def lorentzian(x, x_max=0, sigma=1):
    return 1 / (1 + ((x - x_max) / sigma) ** 2)


def chi_asymmetric(x, x_max=0, sigma_L=1, sigma_R=2, p=2):
    """ Asymmetric filtering function with fast left decay and slow right decay. """
    chi = np.zeros_like(x)
    # Fast decay on the left (exponential)
    chi[x < x_max] = np.exp(-(x_max - x[x < x_max]) / sigma_L)
    
    # Slow decay on the right (power law)
    chi[x >= x_max] = 1 / (1 + ((x[x >= x_max] - x_max) / sigma_R) ** p)
    
    return chi

def closest_value(lst, target):
    return min(lst, key=lambda x: abs(x - target))

def sum_state_vectors(vec1_dict, vec2_dict, N):
    """Returns the full state vector of size 2^N formed by summing two sparse state vectors."""
    dim = 2 ** N
    full_vector = np.zeros(dim, dtype=complex)

    # Generate the Z basis states in the correct order
    z_basis = [''.join(bits) for bits in itertools.product('01', repeat=N)]
    basis_index = {state: i for i, state in enumerate(z_basis)}

    # Add components from first vector
    for state, amp in vec1_dict.items():
        full_vector[basis_index[state]] += amp

    # Add components from second vector
    for state, amp in vec2_dict.items():
        full_vector[basis_index[state]] += amp

    return full_vector


def del_double(liste):
    resultat = []
    vus = set()
    for element in liste:
        if element not in vus:
            resultat.append(element)
            vus.add(element)
    return resultat


def derivee(x, y):
    if len(x) != len(y):
        raise ValueError("Les listes x et y doivent avoir la même longueur.")
    if len(x) < 2:
        raise ValueError("Il faut au moins deux points pour calculer une dérivée.")

    dydx = []
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]
        dydx.append(dy / dx)
    return dydx



 
def get_vector_dict(state_vector,nqubits, threshold):
    z_basis = [''.join(state) for state in itertools.product('01', repeat=nqubits)]
    return {
        z_basis[i]: complex(state_vector[i])
        for i in range(len(state_vector))
        if abs(state_vector[i])**2 > threshold
    }
    
    
def split_list_into_n_sublists(lst, n):
    """
    Splits a list `lst` into `n` sublists, distributing elements as evenly as possible.
    
    Args:
        lst (list): The list to split.
        n (int): Number of sublists to split into.
    
    Returns:
        List[List]: A list of `n` sublists.
    """
    k, r = divmod(len(lst), n)  # k = size of each sublist, r = remainder
    sublists = []
    start = 0
    for i in range(n):
        end = start + k + (1 if i < r else 0)
        sublists.append(lst[start:end])
        start = end
    return sublists