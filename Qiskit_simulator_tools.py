from qiskit import QuantumCircuit
import random
import numpy as np
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate, RZGate, RXGate
import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp, Pauli
from qiskit.circuit.library import UnitaryGate
from scipy.linalg import expm
from typing import Union




def get_expectation_value(circ: QuantumCircuit,obs:str)-> float:
    """get the expectation value of an observable with a given quantum circuit

    Args:
        circ (QuantumCircuit): the quantum circuit to calculate the expectation value from 
        obs (str): The observable to calculate the expectation value from 

    Returns:
        float: The expectation value 
    """
    circ_copy=circ.copy()
    circ_copy.save_expectation_value(SparsePauliOp([obs]), [i for i in range(len(obs))], "0")  # type: ignore
    sim = AerSimulator(method="statevector")
    try:
        data = sim.run(circ_copy).result().data()
    except Exception as e:
        qct = transpile(circ_copy, sim)
        data = sim.run(qct).result().data()
    return data["0"]

def create_plus_state_circuit(nqubits:int)-> QuantumCircuit:
    """creat a circuit that set all the qubits to the + state

    Args:
        nqubits (int): number of qubits

    Returns:
        QuantumCircuit: the quantum circuit that set all the qbits to + state
    """
    circuit = QuantumCircuit(nqubits)
    circuit.h([i for i in range(nqubits)])  
    return circuit

def create_ones_state_circuit(nqubits:int)-> QuantumCircuit:
    """creat a circuit that set all the qubits to the 1 state

    Args:
        nqubits (int): number of qubits

    Returns:
        QuantumCircuit: the quantum circuit that set all the qbits to 1 state
    """
    circuit = QuantumCircuit(nqubits)
    circuit.x([i for i in range(nqubits)])
    return circuit



def rgate(pauli, r):
    return {
        "X": RXGate(r),
        "Z": RZGate(r),
        "XX": RXXGate(r),
        "YY": RYYGate(r),
        "ZZ": RZZGate(r),
    }[pauli]



def gen_quantum_circuit(gates: tuple[str, list[int], float], nq:int, init_state: Union[np.ndarray, list, QuantumCircuit]=None)->QuantumCircuit:
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
    if isinstance(init_state,(np.ndarray, list)):
        circ.initialize(init_state, normalize=True)
        
    for pauli, qubits, coef in gates:
        circ.append(rgate(pauli, coef), qubits)
        
    if  isinstance(init_state,QuantumCircuit):
        circ = init_state.compose(circ,[i for i in range(nq)])
    return circ



def Generate_Evolution_Matrix(hermitian_matrix:np.ndarray):
    """Frome a given hermitian matrix
    generate an evolution matrix as U(t)= exp(-iHt)

    Args:
        hermitian_matrix (np.ndarray): The Hermitian matrix

    Returns:
        function : A function of the time Unitary Gate evolution matrix
    """
    evolution_matrix=(lambda t: UnitaryGate(expm(-1.j*hermitian_matrix*t)))
    return evolution_matrix



def Circuit_to_matrix(circ: QuantumCircuit)-> np.ndarray:
    """Get the matrix representation of a given circuit 

    Args:
        circ (QuantumCircuit): Quantum circuit to get the matrix from

    Returns:
        np.ndarray: the matrix representation of the circuit
    """
    # backend = AerSimulator(method='unitary')
    unitary = Operator(circ).data
    return unitary

def need_transpile(circuit:QuantumCircuit, backend)->bool:
    """return True if a Quantum circuit need to be transpiled before being run

    Args:
        circuit (QuantumCircuit): _description_
        backend (IBMBackend): _description_

    Returns:
        bool: True if the circuit need to be transpiled false if not 
    """
    basis_gates = backend.configuration().basis_gates
    circuit_gates = {op[0].name for op in circuit.data}
    non_basis_gates = circuit_gates - set(basis_gates)
    if len(non_basis_gates)>0:
        return False
    else:
        return True 
    
 
def get_bit_string(circ:QuantumCircuit, shots:int=1, err:list[int,int]=None) ->str:
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
        nm.add_all_qubit_quantum_error(depolarizing_error(err[0], 1), ["x", "z"])
        nm.add_all_qubit_quantum_error(
            depolarizing_error(err[1], 2), ["rzz", "ryy", "rxx"])
        sim = AerSimulator(method="statevector", noise_model=nm)
    else: 
        sim = AerSimulator(method="statevector")
    
    try:
        job_result= sim.run(circ, shots=shots ).result().get_counts()
    except Exception as e:
        job_result=sim.run(transpile(circ,sim), shots=shots ).result()
    finally:
        if shots==1:
            bit_string=str(list(job_result.get_counts().keys())[0])
        else:
            bit_string=job_result.get_counts()
    return bit_string