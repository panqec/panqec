import pytest
from panqec.codes import RotatedPlanar2DCode
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestRotatedPlanar2DCode(StabilizerCodeTest):

    @pytest.fixture(params=[(2, 2), (3, 3), (2, 3)])
    def code(self, request):
        return RotatedPlanar2DCode(*request.param)
