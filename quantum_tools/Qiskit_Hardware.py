# Written by: Hugo PAGES
# Date: 2024-01-05

# Standard library imports
import os
import csv
import datetime
import time
import multiprocessing as mp

# Third-party imports
from tqdm import tqdm
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import SamplerV2, QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeManilaV2


class Hardware:
    """
    A utility class to interface with IBM Quantum hardware via Qiskit's runtime service.

    This class provides methods to:
    - Authenticate using a token or saved account
    - Select a real or fake backend
    - Transpile and run quantum circuits
    - Monitor and retrieve results from jobs
    - Filter available backends based on gate support

    Args:
        token (str, optional): IBM Quantum API token.
        load_account (str, optional): Path to a stored account file to load.
        is_fake (bool, optional): If True, use a simulated backend (e.g., FakeManilaV2).
        set_backend (bool, optional): If True, sets the backend during initialization.
        initiate_service (bool, optional): If True, initiates the QiskitRuntimeService.
    """
    def __init__(
        self,
        token: str = None, #put your token here
        load_account: str = None, # if you want to load an account, put the path of the account to load here 
        is_fake: bool = False, # if True, use a fake backend for testing
        set_backend: bool = True, # if True, set the backend to use for the quantum circuits
        initiate_service:bool= True): # if True, initiate the service with the token or the path of the account: if no token or path given it will 
                                      # try to connect using the default account.
        self.token = token
        self.is_fake = is_fake
        self.load_account = load_account
        
        if initiate_service and not self.is_fake:
            self.__initiate_service(self.token, load_account)
        if set_backend:
            self.set_backend(is_fake)

    def __initiate_service(self, token: str = None, save_account: bool = False, path_account: str = None):
        """
        Initializes a QiskitRuntimeService session.

        Args:
            token (str, optional): IBM Quantum API token.
            save_account (bool): If True, save the current account to disk.
            path_account (str, optional): Path to a previously saved account JSON file.

        Raises:
            Exception: If connection to the service fails.
        """

        if save_account:
            self.save_account(filename="Qiskit_service", set_as_default=False)
       
        channel = ("ibm_quantum")
        connection_failure = False
        
        if isinstance(path_account, str):
            try:
                self.service = QiskitRuntimeService(filename=path_account)
            except Exception as e:
                print(
                    "Error while loading account with name: ", path_account, "error:", e
                )
                connection_failure = True
        if isinstance(token,str):
            try:
                self.service = QiskitRuntimeService(token=token, channel=channel)
            except Exception as e:
                print("Error while loading account with token: ", token, "error:", e)
                connection_failure = True
        else:
            try:
                self.service = QiskitRuntimeService()
            except Exception as e:
                print("Error while loading default account, error:", e)
                connection_failure = True
        if connection_failure:
            print("try to recconect to the service")
        else:
            account = self.service.active_account()
        print("connected to : ", self.service)

    def save_account(self, filename: str = "Qiskit_service", set_as_default: bool = True):
        """
        Saves the currently active IBM Quantum account to a file.

        Args:
            filename (str): File name to store account credentials.
            set_as_default (bool): If True, set this account as the default.
        """
        account = self.service.active_account()
        token = account["token"]
        url = account["url"]
        instance = account["instance"]
        channel = account["channel"]
        self.service.save_account(
            token=token,
            url=url,
            filename=filename,
            instance=instance,
            channel=channel,
            set_as_default=set_as_default,
        )
        print("account saved : in ", filename)

    def set_backend(self, is_fake: bool = True):
        """
            Sets the backend for running quantum circuits.

            Args:
                is_fake (bool): If True, use a fake backend (e.g., FakeManilaV2).
                                If False, select a real hardware backend based on supported gates.
        """
        if is_fake:  # use a fake backend for testing
            self.backend = FakeManilaV2()
        else:  # use a real backend (QPU)
            # self.backend = self.service.backend(name="ibm_strasbourg")
            self.backend = self.service.least_busy()
        print("backend set to : ", self.backend)

    def send_sampler_pub(
        self,
        circuits: list[QuantumCircuit],
        nshots: int = 1,
        verbose: bool = True,
        path_save_id: str = None,
    ) -> tuple[list[str], str]:
        """
        Runs a list of quantum circuits using Qiskit Runtime's Sampler primitive.

        Args:
            circuits (list[QuantumCircuit]): Quantum circuits to execute.
            nshots (int): Number of shots per circuit.
            verbose (bool): If True, print transpilation and job information.
            path_save_id (str): Optional path to save the job ID as a CSV file.

        Returns:
            tuple: (job_ids, None or result) where job_ids is a list of job IDs,
                or the result if using a fake backend.
        """
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        sampler = SamplerV2(self.backend)
        job_id = []
        isa_circuits = []  # list of quantum circuits after transpiling and optimization
        counts = 0
        for n, circ in tqdm(
            enumerate(circuits), desc="transpile circuits", disable=not verbose
        ):
            isa_circuits.append(
                transpile(circ, backend=self.backend, optimization_level=2)
            )
            counts += isa_circuits[-1].size()
        if len(isa_circuits) > 0:
            self.service.check_pending_jobs()
            job = sampler.run(isa_circuits, shots=nshots)
            print("n=", n, "counts=", counts)
            print("number of pub:", len(isa_circuits))
            self.print_job_info(job)
            if isinstance(path_save_id, str):
                self.__save_id(path_save_id, job)
            job_id.append(job.job_id())
        if self.is_fake:
            return job.result()
        return job_id

    def get_sampler_result(self, id):
        """
        Blocks until the specified job finishes and returns the bitstring results.

        Args:
            id (str): Job ID.

        Returns:
            list or str: Measurement results if successful; error message otherwise.
        """
        status = self.get_job_status(id)
        if status == "CANCELLED" or status == "ERROR":
            return f"No results for job : {id}, reason : job {status}"
        t = time.time()
        while status != "DONE":
            print(
                "waiting for job to finish, status :",
                status,
                " waiting time : ",
                time.time() - t,
            )
            time.sleep(10)
            status = self.get_job_status(id)
            if (time.time() - t) / 60 > 30:
                print("Waiting time over 30 min, try later, status : ", status)
                return None
        print("Job finish, status :", status,
              "Total waiting time : ", time.time() - t)
        return self.get_data_from_results(self.get_job_result(id))

    def is_transpiled_for_backend(self, circuit):
        """
        Checks whether a quantum circuit is compatible with the currently set backend.

        Args:
            circuit (QuantumCircuit): Circuit to validate.

        Returns:
            bool: True if compatible; False otherwise.
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
                if (
                    len(instruction.qubits) == 2
                    and instruction.operation.name not in allowed_ops
                ):
                    q1 = circuit.find_bit(instruction.qubits[0]).index
                    q2 = circuit.find_bit(instruction.qubits[1]).index
                    if (q1, q2) not in coupling_map and (q2, q1) not in coupling_map:
                        return False

        return True

    def __save_id(self, path, job):
        filename = os.path.join(path, "job_id.csv")
        with open(filename, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [datetime.datetime.now().strftime(
                    "%Y_%m_%d_%H_%M"), job.job_id() + ""]
            )

    def print_job_info(self, job):
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


    def get_backends_by_basis_gates(self,
        desired_gates: set[str],
        exact_match: bool = False,
        min_qubits: int = None,
        only_operational: bool = True
    ) -> list[str]:
                                    
        """
        Finds available IBMQ backends supporting a specific set of basis gates.

        Args:
            desired_gates (set[str]): Gates required by the user.
            exact_match (bool): If True, match exact set. If False, match if backend supports all desired gates.
            min_qubits (int): Minimum number of qubits.
            only_operational (bool): Return only operational devices.

        Returns:
            list[str]: List of backend names.
        """
        matching = []

        for backend in self.service.backends(simulator=False):
            config = backend.configuration()
            status = backend.status()

            if only_operational and not status.operational:
                continue

            if min_qubits and config.n_qubits < min_qubits:
                continue

            backend_gates = set(config.basis_gates)

            if (exact_match and backend_gates == desired_gates) or \
            (not exact_match and desired_gates.issubset(backend_gates)):
                matching.append(backend.name)

        return matching