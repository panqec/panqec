import pytest
from panqec.codes import Toric2DCode
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestToric2DCode(StabilizerCodeTest):

    @pytest.fixture(params=[(2, 2), (3, 3), (2, 3)])
    def code(self, request):
        return Toric2DCode(*request.param)
