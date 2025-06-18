# Quantum_computing_tools Module

This repository provides a collection of Python utilities for:

- Constructing and serializing Qiskit quantum circuits
- Converting between circuit representations (QASM, matrices, operators)
- Simulating quantum circuits with and without noise
- Generating circuits from gate descriptions
- Performing unitary evolution from Hermitian matrices

It is structured into three core modules: **qiskit tools**, **QASM serialization**, and **mathematical utilities**.

---

## 📁 Modules

### `Qiskit_tools.py` — Qiskit Tools

Utility functions for:

- Creating standard circuits (|1⟩, |+⟩, GHZ-like superpositions)
- Building circuits from gate tuples like `("XX", [0,1], theta)`
- Simulating circuits with and without depolarizing noise
- Getting expectation values using `AerSimulator`
- Checking if a circuit is transpiled for a given backend
- Extracting matrix representations from circuits
- Extracting the last single-qubit gates on each qubit

### `Maths_tools.py` — Mathematical Gate Utilities

Mathematical helpers for gate and matrix operations:

- Generating tensor products of multiple Pauli operators
- Mapping string-labeled gates (e.g. "XX") to `qiskit.circuit.library` gates
- Constructing full unitary matrices from sequences of labeled gates
- Converting Hermitian matrices into time evolution operators `exp(-iHt)`

### `Qasm_tools.py` — QASM Serialization Tools

Tools for working with OpenQASM 2.0:

- Serializing Qiskit circuits to QASM strings
- Inserting custom gate definitions (RXX, RYY, RZZ) into QASM
- Deserializing circuits from QASM using custom gate loaders

---

## 🚀 Installation

Make sure you have the following packages installed:

```bash
pip install qiskit qiskit-aer numpy scipy
```



## License
This project is open-source under the Apache 2.0 License.

