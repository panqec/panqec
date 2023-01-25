import pytest
from panqec.codes import BaconShorCode
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestBaconShorCode(StabilizerCodeTest):

    @pytest.fixture(params=[1, 2, 3])
    def code(self, request):
        return BaconShorCode(request.param)
