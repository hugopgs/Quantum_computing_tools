from Qiskit_Hardware import Hardware
import sys
import os
folder = os.path.dirname(os.path.dirname(
    os.path.abspath(os.path.dirname(sys.argv[0]))))
sys.path.append(folder)

token = ""


def test_initiate_service():
    try:
        Hardware(token)
        assert True
    except Exception as e:
        assert False, f"Error encountered: {e}"


def test_send_sampler_real_backend():
    hardware = Hardware(token)
    from qiskit.circuit.random import random_circuit
    qc = [None]*5
    for i in range(5):
        qc[i] = random_circuit(
            num_qubits=6, depth=5).measure_all(inplace=False)
    try:
        hardware.send_sampler_pub(qc)
        assert True
    except Exception as e:
        assert False, f"Error encountered: {e}"


def test_get_sampler_result():
    hardware = Hardware(token)
    from qiskit.circuit.random import random_circuit
    qc = []
    for _ in range(5):
        qc.append(random_circuit(num_qubits=6,
                  depth=5).measure_all(inplace=False))
    id = hardware.send_sampler_pub(qc, nshots=1)
    res = hardware.get_sampler_result(id)
    assert isinstance(res, list) and len(
        res) == 5, "Error for nshot=1, should return list[str] of length 5 "
    id = hardware.send_sampler_pub(qc, nshots=4000)
    res = hardware.get_sampler_result(id)
    assert isinstance(res, list) and len(res) == 5 and isinstance(res[0], dict)
