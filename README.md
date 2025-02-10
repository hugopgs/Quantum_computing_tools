# Quantum Computing Toolbox

## Overview
This is a list of functions that I often use will doing quantum computing. Some of this function are really easy and dumb but save some times.

## Features
- **Qiskit tools**: Every function that use the Qiskit package, it's only simulation.
- **quantum_toolbox**: Function that can be usefull for manipulating Hamiltonians and their matrix representation.
- **Qiskit Hardware**: Class to facilitate the use of IBM Hardware.


## Qiskit tools
- Generate qiskit QuantumCircuit from list.
- Apply rotation gates (RX, RZ, RXX, RYY, RZZ) to qubits.
- Convert quantum circuits into matrix representations.
- Check if a circuit needs transpilation before execution.
- Create unitary evolution gate for time-dependent simulations.

## quantum_toolbox
- Compute eigenvalues and eigenvectors of Hamiltonians.
- Generate Hermitian matrices with predefined eigenvalues.
- Create unitary evolution matrices for time-dependent simulations.
- Generate Observables
- Find ground_state of Hamiltonian
  
## Qiskit Hardware
- Initialize the IBMRuntimeService
- Send circuits to simulate on real quantum computer 
- Get results from Job Id

## Installation
Ensure you have Python and the required dependencies installed:
```bash
pip install -r requirements.txt
```

## Usage
Import the necessary functions in your script:
```python
from quantum_toolbox import gen_quantum_circuit, get_expectation_value

circuit = gen_quantum_circuit(("X", [0], 1.0), 2)
expectation = get_expectation_value(circuit, "Z")
```



## License
This project is open-source under the Apache 2.0 License.

