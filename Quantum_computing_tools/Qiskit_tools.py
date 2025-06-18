"""
Qiskit Quantum Utilities Module
===============================

This module offers a collection of utility functions and helpers for constructing, analyzing,
simulating, and manipulating quantum circuits using Qiskit. It includes tools for creating
custom quantum circuits, computing expectation values, serializing and deserializing circuits
to and from QASM, generating evolution operators, and simulating noisy measurements.

Key Features:
-------------
- Generation of quantum circuits for specific initial states (|1⟩, |+⟩, superposition).
- Construction of circuits from parameterized gate tuples.
- Simulation of circuits with and without noise using Qiskit Aer.
- Calculation of expectation values from statevector simulation.
- Serialization and deserialization of circuits via OpenQASM 2.0 (including custom gate support).
- Detection of transpilation compatibility with hardware backends.
- Extraction and removal of last single-qubit gates per qubit.
- Matrix representation of quantum circuits via Qiskit `Operator`.

Dependencies:
-------------
- Qiskit: For quantum circuit construction and simulation.
- NumPy: For numerical operations and matrix manipulation.
- SciPy: For computing matrix exponentials.
- Python standard libraries: `typing`, `functools`, `operator`, `itertools`.

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


def qiskit_Generate_Evolution_Matrix(hermitian_matrix: np.ndarray):
    """Frome a given hermitian matrix
    generate an evolution matrix as U(t)= exp(-iHt)

    Args:
        hermitian_matrix (np.ndarray): The Hermitian matrix

    Returns:
        function : A function of the time Unitary Gate evolution matrix
    """
    hamil = (lambda t: UnitaryGate(expm(-1.j*hermitian_matrix*t)))
    return hamil

def serialize_circuit(circuit: QuantumCircuit) -> str:
    """Serialize a QuantumCircuit into QASM."""
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