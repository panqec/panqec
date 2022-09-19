import imp
import numpy as np
from operator import concat
from pymatching import Matching
from panqec.decoders import BaseDecoder
from panqec.codes import Toric2DCode
from panqec.error_models import BaseErrorModel
from panqec.decoders.union_find.clustering import clustering
from panqec.decoders.union_find.peeling import peeling


class UnionFindDecoder(BaseDecoder):
    """Union Find decoder for 2D Toric Code"""

    label = 'Toric 2D Union Find'

    def __init__(self,
                 code: Toric2DCode,
                 error_model: BaseErrorModel,
                 error_rate: float):
        super().__init__(code, error_model, error_rate)

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Get X corrections given code and measured syndrome."""

        # Initialize correction as full bsf.
        correction = np.zeros(2*self.code.n, dtype=np.uint)
        syndromes_z = self.code.extract_z_syndrome(syndrome) # an array with value 1 if that index (of the stabilizer) is a syndrome
        syndromes_x = self.code.extract_x_syndrome(syndrome) # an array with value 1 if that index (of the stabilizer) is a syndrome
        Hz = self.code.Hz # the parity matrix for z stabilizers (a 2D numpy array)
        Hx = self.code.Hx # the parity matrix for z stabilizers (a 2D numpy array) --> is it csr or 2d numpy array? Lynna uses it as csr and it works, Osama uses it as a numpy array and it works!!
        print(f"syndromes_z: {syndromes_z}")
        print(f"syndromes_x: {syndromes_x}")
        print("We are decoding with union find!!!")
        print("\n")

        # clustering
        print("~~~~ we do clustering now ~~~~")
        output_x = clustering(syndromes_z, Hz)
        output_z = clustering(syndromes_x, Hx)
        print(f"The output_x of clustering is: {output_x}")
        print(f"The output_z of clustering is: {output_z}")
        print("\n")  

        # peeling
        print("~~~~ we start peeling for x_correction ~~~~")
        correction_x = peeling(output_x, syndromes_z, Hz)
        print(f"~~ correction_x is: {correction_x} ~~")
        print("\n \n")
        print("~~~~ we start peeling for z_correction ~~~~")
        correction_z = peeling(output_z, syndromes_x, Hx)
        print(f"~~ correction_z is: {correction_z} ~~")

        # Load the correction into the X block of the full bsf ###
        correction[:self.code.n] = correction_x
        correction[self.code.n:] = correction_z        

        print("~~~ Decoding Over ~~~")
        return correction


