"""
Quantum Tools Module
====================

This module provides a collection of tools and utility functions for the simulation and manipulation of quantum circuits, primarily using the Qiskit framework. The functionalities include creating quantum circuits with specific states (|1> state and |+> state), computing expectation values of observables for a given circuit, generating Hermitian and unitary matrices, and various operations on matrices and quantum state vectors.

Key Features:
- Creation of quantum circuits with specific states (|1> state and |+> state).
- Computation of expectation values of observables for a given circuit.
- Simulation of quantum circuits with or without noise.
- Generation of Hermitian and unitary matrices from specified eigenvalues.
- Calculation of eigenenergies and energy gaps of a Hamiltonian.
- Determination of the ground state of a Hamiltonian.

This module is designed for researchers and developers working in the field of quantum computing and quantum simulation.

Requirements:
- Qiskit: Main library for quantum simulations.
- NumPy: Used for matrix and vector operations.

Author: Hugo PAGES, hugo.pages@etu.unistra.fr
Date: [12/02/2025]
"""

from qiskit import QuantumCircuit
import random
import numpy as np
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate, RZGate, RXGate
from qiskit.quantum_info import Operator, SparsePauliOp, Pauli
from qiskit.circuit.library import UnitaryGate
from scipy.linalg import expm
from typing import Union
from functools import reduce
from operator import concat
import itertools


def qiskit_rgate(pauli, r):
    return {
        "X": RXGate(r),
        "Z": RZGate(r),
        "XX": RXXGate(r),
        "YY": RYYGate(r),
        "ZZ": RZZGate(r),
    }[pauli]


def qiskit_get_expectation_value(self, circ: Union[QuantumCircuit, str], obs: str) -> float:
    if isinstance(circ, str):
        circ= self.deserialize_circuit(circ)
    circ_copy = circ.copy()
    circ_copy.save_expectation_value(SparsePauliOp(
        [obs]), [i for i in range(len(obs))], "0")  # type: ignore
    sim = AerSimulator(method="statevector")
    try:
        data = sim.run(circ_copy).result().data()
    except Exception as e:
        qct = transpile(circ_copy, sim)
        data = sim.run(qct).result().data()
    return data["0"]    



def qiskit_sampler(circ: QuantumCircuit, shots: int = 1, err: list[int, int] = None, get_bit_string:bool=True) -> str:
    """bit string measurement of a given quantum circuit.

    Args:
        circ (QuantumCircuit): quantum circuit to measure
        shots (int, optional): number of shots. Defaults to 1.
        err (list[int,int], optional): the depolarizing error, err[0] for x and z gates and err[1] for rxx,ryy and rzz gate. Defaults to None.
    Returns:
        str: bit string 
    """
    if isinstance(err, (list, np.ndarray)):
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(
            depolarizing_error(err[0], 1), ["x", "z"])
        nm.add_all_qubit_quantum_error(
            depolarizing_error(err[1], 2), ["rzz", "ryy", "rxx"])
        sim = AerSimulator(method="statevector", noise_model=nm)
    else:
        sim = AerSimulator(method="statevector")

    try:
        job_result = sim.run(circ, shots=shots).result()
    except Exception as e:
        job_result = sim.run(transpile(circ, sim), shots=shots).result()
    finally:
        if get_bit_string:
            if shots == 1:
                bit_string = str(list(job_result.get_counts().keys())[0])
            else:
                bit_string = job_result.get_counts()
            return bit_string
        else: 
            return job_result

def qiskit_create_ones_state_circuit(nqubits: int) -> QuantumCircuit:
    """creat a circuit that set all the qubits to the 1 state

    Args:
        nqubits (int): number of qubits
    Returns:
        QuantumCircuit: the quantum circuit that set all the qbits to 1 state
    """
    circuit = QuantumCircuit(nqubits)
    circuit.x([i for i in range(nqubits)])
    return circuit


def qiskit_create_superposition_circuit(numQs: int) -> QuantumCircuit:
    """
    Creat a quantum circuit to obtain a superposition of state: |000...0> + |111...1>.
    Args:
        numQs (int): Number of qubits.
    Returns:
        QuantumCircuit
    """
    qc = QuantumCircuit(numQs)
    qc.h(0)
    for i in range(numQs - 1):
        qc.cx(i, i + 1)
    return qc


def qiskit_create_plus_state_circuit(nqubits: int) -> QuantumCircuit:
    """creat a circuit that set all the qubits to the + state

    Args:
        nqubits (int): number of qubits

    Returns:
        QuantumCircuit: the quantum circuit that set all the qbits to + state
    """
    circuit = QuantumCircuit(nqubits)
    circuit.h([i for i in range(nqubits)])
    return circuit


def qikist_gen_quantum_circuit(gates: tuple[str, list[int], float], nq: int, init_state: Union[np.ndarray, list, QuantumCircuit] = None) -> QuantumCircuit:
    """Generate a quantum circuite given a list of gates.

    Args:
        gates (tuple[str, list[int], float]): The list of gates to generate the circuit from, in the form of : ("Gate", [nq1, nq2], parameters)
        exemple: ("XX",[2,3], 1) gate rxx on qubit 2, 3 with parameter 1
        nq (int): total number of qubit
        init_state (QuantumCircuit or np.ndarray, optional): Initialize the Quantum circuit, if its a quantum circuit 
        it will add the quantum circuit at the begining if it's an array initialize the quantum state using the initialize method 
        to put at the beginning of the circuit. Defaults to None.

    Returns:
        QuantumCircuit: The quantum circuit representation of the given gates
    """
    circ = QuantumCircuit(nq)
    if isinstance(init_state, (np.ndarray, list)):
        circ.initialize(init_state, normalize=True)
    for pauli, qubits, coef in gates:
        circ.append(qiskit_rgate(pauli, 2*coef), qubits)
    if isinstance(init_state, QuantumCircuit):
        circ = init_state.compose(circ, [i for i in range(nq)])
    return circ


def qiskit_is_transpiled_for_backend(circuit, backend):
    """
    Check if a circuit appears to be transpiled for a specific backend.

    Args:
        circuit (QuantumCircuit): The circuit to check
        backend (Backend): The backend to check against

    Returns:
        bool: True if circuit appears to be transpiled for this backend
    """
    backend_config = backend.configuration()
    basis_gates = backend_config.basis_gates
    allowed_ops = ["barrier", "snapshot", "measure", "reset"]
    for instruction in circuit.data:
        gate_name = instruction.operation.name
        if gate_name not in basis_gates and gate_name not in allowed_ops:
            return False
    coupling_map = getattr(backend_config, "coupling_map", None)
    if coupling_map:
        # Convert coupling map to list of tuples if it's not already
        if not isinstance(coupling_map[0], tuple):
            coupling_map = [(i, j) for i, j in coupling_map]
        # Check each 2-qubit gate (excluding measurement operations)
        for instruction in circuit.data:
            if len(instruction.qubits) == 2 and instruction.operation.name not in allowed_ops:
                q1 = circuit.find_bit(instruction.qubits[0]).index
                q2 = circuit.find_bit(instruction.qubits[1]).index
                if (q1, q2) not in coupling_map and (q2, q1) not in coupling_map:
                    return False

    return True


def serialize_circuit(circuit: QuantumCircuit) -> str:
    """Serialize a QuantumCircuit into JSON."""
    from qiskit.qasm2 import dumps
    qasm_string = dumps(circuit)  # Convert circuit to OpenQASM 2.0
    gate_definitions = """
    gate rxx(theta) a, b { cx a, b; rz(theta) b; cx a, b; }
    gate ryy(theta) a, b { ry(-pi/2) a; ry(-pi/2) b; cx a, b; rz(theta) b; cx a, b; ry(pi/2) a; ry(pi/2) b; }
    gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }
    """
    qasm_lines = qasm_string.split("\n")
    qasm_lines.insert(2, gate_definitions.strip())
    qasm = "".join(qasm_lines)
    return qasm


def deserialize_circuit(qasm_str, custom_instructions:list=None):
    # VERY UNSTABLE, Lot of gates not recognize->prefere clifford circuit without swap gate
    """Deserialize a QuantumCircuit from JSON."""
    from qiskit.qasm2 import loads, CustomInstruction
    rxx_custom = CustomInstruction(
        name="rxx", num_params=1, num_qubits=2, builtin=False, constructor=RXXGate)
    ryy_custom = CustomInstruction(
        name="ryy", num_params=1, num_qubits=2, builtin=False,  constructor=RYYGate)
    rzz_custom = CustomInstruction(
        name="rzz", num_params=1, num_qubits=2, builtin=False, constructor=RZZGate)
    custom_instruction_list=[rxx_custom,ryy_custom,rzz_custom]
    if isinstance(custom_instructions, list):
        for instruction in custom_instructions:
            custom_instruction_list.append(instruction)
        
    return loads(qasm_str, custom_instructions=custom_instruction_list)



def count_qubits_in_qasm(qasm_string):
    """Count the total number of qubits declared in a QASM file."""
    import re
    
    # Find all qreg declarations using regex
    qreg_pattern = r'qreg\s+([a-zA-Z0-9_]+)\[(\d+)\];'
    qreg_matches = re.findall(qreg_pattern, qasm_string)
    
    # Sum up the sizes of all quantum registers
    total_qubits = sum(int(size) for _, size in qreg_matches)
    
    return total_qubits

def generate_qasm_from_gates(gates, num_qubits):
    """
    Generates an OpenQASM string directly from a list of (pauli, qubits, coef) tuples.

    Args:
        gates (list of tuples): List containing (pauli, qubits, coef).
        num_qubits (int): Number of qubits in the quantum circuit.

    Returns:
        str: OpenQASM 2.0 string.
    """
    # Start the QASM string
    qasm = f"OPENQASM 2.0;\ninclude \"qelib1.inc\";\n"

    # Define the rotation gate mapping
    gate_map = {
        "X": "rx({})",
        "Z": "rz({})",
        "XX": "rxx({})",
        "YY": "ryy({})",
        "ZZ": "rzz({})"
    }

    # Declare the quantum register
    qasm += f"qreg q[{num_qubits}];\n"

    # Apply each gate in the list and append to the QASM string
    for pauli, qubits, coef in gates:
        if pauli in gate_map:
            gate_str = gate_map[pauli].format(coef)
            qasm += f"{gate_str} q[{qubits[0]}], q[{qubits[1]}];\n" if len(
                qubits) == 2 else f"{gate_str} q[{qubits[0]}];\n"
        else:
            raise ValueError(f"Unsupported gate type: {pauli}")

    return qasm


def qiskit_get_last_single_qubit_gates(circuit: QuantumCircuit):
    last_single_qubit_gate_indices = [-1] * circuit.num_qubits
    for index in range(len(circuit.data) - 1, -1, -1):
        instruction = circuit.data[index]
        op = instruction.operation
        qargs = instruction.qubits

        if len(qargs) == 1:
            qubit_index = circuit.qubits.index(qargs[0])
            if last_single_qubit_gate_indices[qubit_index] == -1:
                last_single_qubit_gate_indices[qubit_index] = index

    data = [instruction for i, instruction in enumerate(
        circuit.data) if i in last_single_qubit_gate_indices]
    return data


def remove_last_single_qubit_gates(self, circuit):
    # Initialize a list to track the last gate indices for each qubit
    last_single_qubit_gate_indices = [-1] * circuit.num_qubits

    # Traverse the circuit in reverse to find the last single-qubit gate for each qubit
    for index in range(len(circuit.data) - 1, -1, -1):
        instruction = circuit.data[index]
        op = instruction.operation
        qargs = instruction.qubits

        if len(qargs) == 1:  # Check if it's a single-qubit gate
            # Get index of the qubit in the circuit
            qubit_index = circuit.qubits.index(qargs[0])

            # If this is the first last single-qubit gate encountered, record its position
            if last_single_qubit_gate_indices[qubit_index] == -1:
                last_single_qubit_gate_indices[qubit_index] = index

    # Filter the data to exclude the last single-qubit gate for each qubit
    new_data = [instruction for i, instruction in enumerate(
        circuit.data) if i not in last_single_qubit_gate_indices]
    circuit.data = new_data

def qiskit_circuit_to_matrix(circ: QuantumCircuit) -> np.ndarray:
    """Get the matrix representation of a given circuit 

    Args:
        circ (QuantumCircuit): Quantum circuit to get the matrix from

    Returns:
        np.ndarray: the matrix representation of the circuit
    """
    # backend = AerSimulator(method='unitary')
    unitary = Operator(circ).data
    return unitary


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


def Generate_Evolution_Matrix(hermitian_matrix: np.ndarray):
    """Frome a given hermitian matrix
    generate an evolution matrix as U(t)= exp(-iHt)

    Args:
        hermitian_matrix (np.ndarray): The Hermitian matrix

    Returns:
        function : A function of the time Unitary Gate evolution matrix
    """
    hamil = (lambda t: UnitaryGate(expm(-1.j*hermitian_matrix*t)))
    return hamil


def get_expectation_value(circ: QuantumCircuit, obs: str) -> float:
    """get the expectation value of an observable with a given quantum circuit

    Args:
        circ (QuantumCircuit): the quantum circuit to calculate the expectation value from 
        obs (str): The observable to calculate the expectation value from 

    Returns:
        float: The expectation value 
    """
    circ_copy = circ.copy()
    circ_copy.save_expectation_value(SparsePauliOp(
        [obs]), [i for i in range(len(obs))], "0")  # type: ignore
    sim = AerSimulator(method="statevector")
    try:
        data = sim.run(circ_copy).result().data()
    except Exception as e:
        qct = transpile(circ_copy, sim)
        data = sim.run(qct).result().data()
    return data["0"]


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