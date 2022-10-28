import pytest
from typing import List
from abc import ABCMeta, abstractmethod
import numpy as np
from panqec.codes import StabilizerCode
from panqec.decoders import BaseDecoder
from panqec.error_models import BaseErrorModel, PauliErrorModel


class DecoderTest(metaclass=ABCMeta):

    @pytest.fixture
    @abstractmethod
    def code(self) -> StabilizerCode:
        pass

    @pytest.fixture
    def error_model(self) -> BaseErrorModel:
        return PauliErrorModel(1/3, 1/3, 1/3)

    @pytest.fixture
    @abstractmethod
    def decoder(self, code, error_model) -> BaseDecoder:
        pass

    @pytest.fixture
    def allowed_paulis(self) -> List[str]:
        return ['X', 'Y', 'Z']

    def test_decode_trivial_syndrome(self, code, decoder):
        syndrome = np.zeros(
            shape=code.stabilizer_matrix.shape[0], dtype=np.uint
        )
        correction = decoder.decode(syndrome)
        assert correction.shape[0] == 2*code.n
        assert np.all(code.measure_syndrome(correction) == 0)
        assert np.all(correction == 0)
        assert code.is_success(correction)
        assert code.in_codespace(correction)

    @pytest.mark.slow
    def test_decode_single_qubit_error(self, code, decoder, allowed_paulis):
        for pauli in allowed_paulis:
            for i in range(code.n):
                error = np.zeros(2*code.n, dtype='uint8')

                if pauli in ['X', 'Y']:
                    error[i] = 1
                if pauli in ['Z', 'Y']:
                    error[code.n + i] = 1

                syndrome = code.measure_syndrome(error)

                correction = decoder.decode(syndrome)
                total_error = (correction + error) % 2

                assert code.in_codespace(total_error), (
                    "The decoding result is outside of the codespace"
                    f"for single-qubit {pauli} error at qubit {i}"
                )

                assert code.is_success(total_error), (
                    "The decoding result is has created a logical"
                    f"for single-qubit {pauli} error at qubit {i}"
                )
