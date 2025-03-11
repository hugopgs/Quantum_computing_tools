from qiskit import QuantumCircuit
import numpy as np
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate, RZGate, RXGate
from qiskit.quantum_info import Operator, SparsePauliOp, Pauli
from typing import Union
from qiskit.circuit.library import UnitaryGate
from scipy.linalg import expm


def qiskit_rgate(pauli, r):
    return {
        "X": RXGate(r),
        "Z": RZGate(r),
        "XX": RXXGate(r),
        "YY": RYYGate(r),
        "ZZ": RZZGate(r),
    }[pauli]



def qiskit_get_expectation_value(circ: QuantumCircuit, obs: str) -> float:
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


def qiskit_sampler(circ: QuantumCircuit, shots: int = 1, err: list[int, int] = None) -> str:
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
        job_result = sim.run(circ, shots=shots).result().get_counts()
    except Exception as e:
        job_result = sim.run(transpile(circ, sim), shots=shots).result()
    finally:
        if shots == 1:
            bit_string = str(list(job_result.get_counts().keys())[0])
        else:
            bit_string = job_result.get_counts()
    return bit_string


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


def deserialize_circuit(qasm_str):
    # VERY UNSTABLE, Lot of gates not recognize->prefere clifford circuit without swap gate
    """Deserialize a QuantumCircuit from JSON."""
    from qiskit.qasm2 import loads, CustomInstruction
    # Define custom instructions for rxx, ryy, and rzz
    rxx_custom = CustomInstruction(
        name="rxx", num_params=1, num_qubits=2, builtin=False, constructor=RXXGate)
    ryy_custom = CustomInstruction(
        name="ryy", num_params=1, num_qubits=2, builtin=False,  constructor=RYYGate)
    rzz_custom = CustomInstruction(
        name="rzz", num_params=1, num_qubits=2, builtin=False, constructor=RZZGate)
    return loads(qasm_str, custom_instructions=[rxx_custom, ryy_custom, rzz_custom])


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


def qiskit_remove_last_single_qubit_gates(circuit: QuantumCircuit):
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
