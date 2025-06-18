"""
QASM Tools Module
====================

This module provides utility functions for quantum circuit construction, analysis, and manipulation,
primarily using QASM .

Main Features
-------------
- Count the number of qubits from a QASM string.
- Generate OpenQASM 2.0 code from a list of parameterized gates.
- Compute the depth of a quantum circuit from QASM using supported gates (rx, rz, rxx, ryy, rzz).
- Remove the last single-qubit gate on each qubit from a Qiskit circuit.

Dependencies
------------
- NumPy
- re (Python standard library)

Author
------
Hugo Pages  
Email: hugo.pages@etu.unistra.fr  
Date: 12/02/2025
"""


import numpy as np


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

import re

def get_depth_from_qasm(qasm_str: str) -> int:
    """
    Compute the circuit depth from a QASM string using only rx, rz, rxx, ryy, rzz gates.

    Args:
        qasm_str (str): The QASM string.

    Returns:
        int: The circuit depth.
    """
    if isinstance(qasm_str, list):
        depth_list = []
        for i in range(len(qasm_str)):
            depth_list.append(get_depth_from_qasm(qasm_str[i]))
        return np.mean(depth_list)
    # Match the qreg definition
    qreg_match = re.search(r"qreg\s+(\w+)\[(\d+)\];", qasm_str)
    if not qreg_match:
        raise ValueError("No qreg declaration found.")

    prefix, size = qreg_match.group(1), int(qreg_match.group(2))
    qubit_depths = [0] * size
    max_depth = 0

    # Only consider these gates
    gate_pattern = re.compile(
        r"(rx|rz|rxx|ryy|rzz)\s*\([^)]*\)\s+([a-zA-Z_]+\[\d+\](?:\s*,\s*[a-zA-Z_]+\[\d+\])?)\s*;",
        re.IGNORECASE
    )

    for match in gate_pattern.finditer(qasm_str):
        gate = match.group(1).lower()
        qubit_args = match.group(2).replace(" ", "")
        qubits = [int(q.split('[')[1][:-1]) for q in qubit_args.split(",")]

        current_layer = max(qubit_depths[q] for q in qubits) + 1

        for q in qubits:
            qubit_depths[q] = current_layer

        max_depth = max(max_depth, current_layer)

    return max_depth





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


