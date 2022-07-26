import pytest
from panqec.codes import BaconShorCode
from panqec.bpauli import bsf_to_pauli


class Test2x2BaconShor:

    @pytest.fixture
    def code(self):
        return BaconShorCode(2, 2)

    def test_gauge_operators(self, code):
        gauges = [
            bsf_to_pauli(bsf)[0] for bsf in code.gauge_matrix
        ]
        assert gauges == ['XIXI', 'IXIX', 'ZZII', 'IIZZ']

    def test_initialize_bacon_shor_code(self, code):
        stabilizers = [
            bsf_to_pauli(bsf)[0] for bsf in code.stabilizer_matrix
        ]
        assert stabilizers == ['XXXX', 'ZZZZ']
