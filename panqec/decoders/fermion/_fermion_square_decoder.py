import numpy as np
from panqec.codes import StabilizerCode
from panqec.error_models import BaseErrorModel
from panqec.decoders import BaseDecoder
import panqec.bsparse as bsparse

class FermionSquareDecoder(BaseDecoder):
    """Basic Fermion Square Encoding Decoder:
    Searches for isolated 1-qubit error signatures for X,Y,Z errors on face qubits
    and X,Y errors (have the same signature) on vertex qubits. Z-errors on vertex
    qubits are undetectable and are ignored as a natural phase noise."""

    label = 'Fermion Square Decoder'
    allowed_codes = ['FermionSquare']
        
    #get coordinates of x,y,z errors separately
    def error_coordinates(self, errors):
        x_errors=[]
        y_errors=[]
        z_errors=[]
        for errors_index in range(len(errors)):
            if errors_index < len(self.code.qubit_coordinates) and errors[errors_index]:
                x_errors.append(self.code.qubit_coordinates[errors_index])
            elif errors[errors_index]:
                if self.code.qubit_coordinates[errors_index%len(self.code.qubit_coordinates)] in x_errors:
                    y_errors.append(self.code.qubit_coordinates[errors_index%len(self.code.qubit_coordinates)])
                    x_errors.remove(self.code.qubit_coordinates[errors_index%len(self.code.qubit_coordinates)])
                else:
                    z_errors.append(self.code.qubit_coordinates[errors_index%len(self.code.qubit_coordinates)])
        return x_errors, y_errors, z_errors

    
    #get coordinates of activated stabilizers
    def stabilizer_coordinates(self, syndrome):
        stabilizers = []
        for stab_ind in range(len(syndrome)):
            if syndrome[stab_ind]:
                stabilizers.append(self.code.stabilizer_coordinates[stab_ind])
        return stabilizers
    
 
    #look for face qubit error patterns, return correction and updated syndrome
    def decode(self, code: StabilizerCode, syndrome: np.ndarray, **kwargs) -> np.ndarray:
        """Get X,Z corrections given code and measured syndrome."""
        self.code = code
        correction = np.zeros(2*self.code.n, dtype=np.uint)
        updated_syndrome = copy.deepcopy(syndrome)
        key_list = list(self.code.qsmap.keys())
        num_vertex_qubits = code.size[0]**2
        key_list_vertex = key_list[:num_vertex_qubits]
        random.shuffle(key_list_vertex)
        key_list_face = key_list[num_vertex_qubits:]
        random.shuffle(key_list_face)
        random.shuffle(key_list)
        
        for q_i in key_list_face + key_list_vertex: #face first, then vertex, both shuffled separately
        #for q_i in key_list: # all shuffled
            corrected = False
            stab_arr = self.code.qsmap[q_i]
            #face qubit
            if len(stab_arr) == 4:
                if updated_syndrome[stab_arr[0]] and updated_syndrome[stab_arr[1]] and updated_syndrome[stab_arr[2]] and updated_syndrome[stab_arr[3]]:
                    #Z error
                    correction[q_i + self.code.n] = (correction[q_i + self.code.n] + 1) % 2
                    corrected = True
                elif updated_syndrome[stab_arr[0]] and updated_syndrome[stab_arr[1]]:
                    #X error
                    correction[q_i] = (correction[q_i] + 1) % 2
                    corrected = True
                elif updated_syndrome[stab_arr[2]] and updated_syndrome[stab_arr[3]]:
                    #Y error
                    correction[q_i] = (correction[q_i] + 1) % 2
                    correction[q_i + self.code.n] = (correction[q_i + self.code.n] + 1) % 2
                    corrected = True
            #vertex qubit
            elif updated_syndrome[stab_arr[0]] and updated_syndrome[stab_arr[1]]:
                #X or Y error - random correction
                coin = random.randint(0,1)
                if coin:
                    correction[q_i] = (correction[q_i] + 1) % 2
                else:
                    correction[q_i] = (correction[q_i] + 1) % 2
                    correction[q_i + self.code.n] = (correction[q_i + self.code.n] + 1) % 2       
                corrected = True
            if corrected:
                for stab in stab_arr:
                    updated_syndrome[stab] = 0
        #if syndrome still contains anything, decoding failed, return 
        if any(updated_syndrome):
            return bsparse.from_array([2])
        return bsparse.from_array(correction)