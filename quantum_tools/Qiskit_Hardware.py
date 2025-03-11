from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit import QuantumCircuit, transpile
import os
import csv
import datetime
import time
from tqdm import tqdm
import multiprocessing as mp


class Hardware():
    def __init__(self, token: str, connect: bool = True, Fake_backend: bool = False, name_backend: str = None):
        self.token = token
        self.Fake_backend = Fake_backend
        self.__initiate_service(self.token)
        if connect:
            self.set_backend(Fake_backend, name_backend)
            print("connected to : ", self.backend)

    def __initiate_service(self, token: str) -> QiskitRuntimeService:
        """
        Initiates the service with the provided token.
        """
        self.service = QiskitRuntimeService(
            channel='ibm_quantum',
            instance='ibm-q/open/main',
            token=token
        )
        self.service.check_pending_jobs()

    def set_backend(self, fake: bool = False, name: str = None):
        if fake:
            self.backend = FakeManilaV2()
            self.Fake_backend = True
        else:
            if isinstance(name, str):
                self.backend = self.service.backend(name)
            else:
                self.backend = self.service.least_busy(
                    operational=True, simulator=False, min_num_qubits=3)
        self.hardware_name = self.backend.name

    def send_sampler_pub(self, circuits: list[QuantumCircuit], nshots: int = 1, verbose: bool = True, path_save_id: str = None) -> tuple[list[str], str]:
        """
        Measure the bit string of the given quantum circuits on IBM's quantum devices.

        Args:
            service (QiskitRuntimeService): IBM quantum service
            circuits (list[QuantumCircuit]): List of quantum circuits to measure
            verbose (bool, optional): If True, print job informations. Defaults to True.
            get_id_only (bool, optional): If True, return only the job id. Defaults to False.
            path (str, optional): Path to save the job id. Defaults to None.
            fake (bool, optional): If True, use a fake backend for testing. Defaults to False.
        Returns:
            list[str]: Bit string array
            str: Job id
            if get_id_only== True : return [], job.job_id() 

        """
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        sampler = SamplerV2(self.backend)
        job_id = []
        isa_circuits = []  # list of quantum circuits after transpiling and optimization
        counts = 0
        for n, circ in tqdm(enumerate(circuits), desc="transpile circuits", disable=not verbose):
            isa_circuits.append(
                transpile(circ, backend=self.backend, optimization_level=2))
            counts += isa_circuits[-1].size()
            if counts > 19_000_000 and not(self.Fake_backend):
                print("n=", n, "counts=", counts)
                print("number of pub:", len(isa_circuits))
                counts = 0
                self.service.check_pending_jobs()
                job = sampler.run(isa_circuits, shots=nshots)
                self.__print_job_info(job)
                isa_circuits = []
                job_id.append(job.job_id())
                if isinstance(path_save_id, str):
                    self.__save_id(path_save_id, job)

        if len(isa_circuits) > 0:
            self.service.check_pending_jobs()
            job = sampler.run(isa_circuits, shots=nshots)
            print("n=", n, "counts=", counts)
            print("number of pub:", len(isa_circuits))
            self.__print_job_info(job)
            if isinstance(path_save_id, str):
                self.__save_id(path_save_id, job)
            job_id.append(job.job_id())
        if self.Fake_backend:
            return job.result()
        return job_id

    def get_sampler_result(self, id):
        status = self.get_job_status(id)
        if status == "CANCELLED" or status == "ERROR":
            return f"No results for job : {id}, reason : job {status}"
        t = time.time()
        while status != "DONE":
            print("waiting for job to finish, status :",
                  status, " waiting time : ", time.time()-t)
            time.sleep(10)
            status = self.get_job_status(id)
            if (time.time()-t)/60 > 30:
                print("Waiting time over 30 min, try later, status : ", status)
                return None
        print("Job finish, status :",  status, "Total waiting time : ", time.time()-t)
        return self.get_data_from_results(self.get_job_result(id))
        

    def is_transpiled_for_backend(self, circuit):
        """
        Check if a circuit appears to be transpiled for a specific backend.

        Args:
            circuit (QuantumCircuit): The circuit to check
            backend (Backend): The backend to check against

        Returns:
            bool: True if circuit appears to be transpiled for this backend
        """
        # Get the backend's configuration
        backend_config = self.backend.configuration()
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

    def __save_id(self, path, job):
        filename = os.path.join(path, "job_id.csv")
        with open(filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.datetime.now().strftime(
                "%Y_%m_%d_%H_%M"), job.job_id()+''])

    def __print_job_info(self, job):
        print(f">>> Running on {self.backend.name}")
        print(f">>> Job ID: {job.job_id()}")
        print(f">>> Job Status: {job.status()}")

    def get_job_status(self, id):
        job = self.service.job(id)
        return job.status()

    def get_job_result(self, id):
        job = self.service.job(id)
        result = job.result()
        return result

    def get_data_from_results(self, results):
        bit_string_array = []
        for pub in results:
            if pub.data.meas.num_shots == 1:
                bit_string = str(list(pub.data.meas.get_counts().keys())[0])
            else:
                bit_string = pub.data.meas.get_counts()
            bit_string_array.append(bit_string)
        return bit_string_array