import pytest
from panqec.codes import ChamonCode
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestChamonCode(StabilizerCodeTest):

    @pytest.fixture(params=[(3, 2, 2), (5, 4, 4)])
    def code(self, request):
        if isinstance(request.param, tuple):
            return ChamonCode(*request.param)
        else:
            return ChamonCode(request.param)
