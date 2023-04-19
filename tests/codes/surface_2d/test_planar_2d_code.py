import pytest
from panqec.codes import Planar2DCode
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestPlanar2DCode(StabilizerCodeTest):

    @pytest.fixture(params=[(2, 2), (3, 3), (2, 3)])
    def code(self, request):
        return Planar2DCode(*request.param)
