import pytest
import numpy as np
from qecsim.models.toric import ToricCode
from qecsim.paulitools import bsf_wt
from bn3d.decoders import Toric2DPymatchingDecoder
from bn3d.bpauli import bcommute
from bn3d.bsparse import to_array


@pytest.fixture
def code():
    return ToricCode(4, 5)


class TestToric2DPymatchingDecoder:

    @pytest.mark.parametrize(
        'operator, location',
        [
            ('X', (0, 1, 1)),
            ('Y', (0, 1, 1)),
            ('Z', (0, 1, 1)),
            ('X', (1, 1, 1)),
            ('Y', (1, 1, 1)),
            ('Z', (1, 1, 1)),
        ]
    )
    def test_decode_single_error(self, code, operator, location):
        decoder = Toric2DPymatchingDecoder()
        pauli = code.new_pauli()
        pauli.site(operator, location)
        error = pauli.to_bsf()
        assert bsf_wt(error) == 1, 'Error should be weight 1'
        syndromes = bcommute(code.stabilizers, error)
        if operator in ['Z', 'X']:
            assert sum(syndromes) == 2, 'Should be 2 syndromes if X or Z error'
        elif operator == 'Y':
            assert sum(syndromes) == 4, 'Should be 4 syndromes if Y error'
        correction = decoder.decode(code, syndromes)
        assert bsf_wt(correction) == 1, 'Correction should be weight 1'
        total_error = to_array(error + correction) % 2
        assert np.all(bcommute(code.stabilizers, total_error) == 0), (
            'Total error should be in code space'
        )
