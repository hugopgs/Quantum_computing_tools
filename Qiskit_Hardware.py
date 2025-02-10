from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit import QuantumCircuit, transpile
import os, csv, datetime
from tqdm import tqdm


class Hardware(): 
    def __init__(self, token: str, Fake_backend:bool=False, name_backend:str=None):
        self.token= token 
        self.Fake_backend=Fake_backend
        self.__initiate_service(self.token)
        self.__set_backend(Fake_backend,name_backend)
    
        
    def __initiate_service(self, token: str)-> QiskitRuntimeService :
        """
        Initiates the service with the provided token.
        """
        self.service= QiskitRuntimeService(
            channel='ibm_quantum',
            instance='ibm-q/open/main',
            token=token
        )
        
        
    def __set_backend(self, fake:bool= False, name:str=None):
            if fake:
                self.backend= FakeManilaV2()
            else :
                if isinstance(name, str):
                    self.backend= self.service.backend(name)
                else:
                    self.backend =self.service.least_busy(operational=True, simulator=False, min_num_qubits=3)
            self.hardware_name=self.backend.name
    
    
    def get_job_result(self, id):
        job = self.service.job(id)
        result=job.result()
        return result
    
    
    def send_sampler_pub(self,circuits: list[QuantumCircuit], nshots:int=1,verbose= True, path:str=None)->tuple[list[str], str]:
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
            circuits=[circuits]
            
            
        sampler = SamplerV2(self.backend)
        isa_circuits=[] # list of quantum circuits after transpiling and optimization
        for circuit in tqdm(circuits, desc="transpil circuits"):
            isa_circuit= transpile(circuit, backend= self.backend, optimization_level=2)
            isa_circuits.append(isa_circuit)
        job=sampler.run(isa_circuits,shots=nshots)
        if verbose: # print job informations
            print(f"number of pubs: {len(circuits)}")    
            print(f">>> Running on {self.backend.name}")
            print(f">>> Job ID: {job.job_id()}")
            print(f">>> Job Status: {job.status()}")
            if isinstance(path,  str):
                filename = os.path.join(path, "job_id.csv")    
                with open(filename, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"), job.job_id()+''])
        if self.Fake_backend:
            return job.result()   
                    
        return job.job_id()


    def get_sampler_result(self,id):
        result=self.get_job_result(id)
        bit_string_array=[]
        for pub in result:
            if pub.data.meas.num_shots==1:
                bit_string=str(list(pub.data.meas.get_counts().keys())[0])
            else:
                bit_string=pub.data.meas.get_counts()
            bit_string_array.append(bit_string)
        return bit_string_array
    
    
    def test_hardware(self):
        qc=QuantumCircuit(2)
        qc.h(0)
        qc.cx(0,1)
        qc.measure_all()
        id=self.send_sampler_pub(qc)
        print(self.get_sampler_result(id))
        