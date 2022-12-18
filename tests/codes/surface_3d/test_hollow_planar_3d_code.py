import pytest
from panqec.codes import HollowPlanar3DCode
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestHollowPlanar3DCode(StabilizerCodeTest):

    @pytest.fixture(params=[(2, 2, 2), (3, 3, 3), (2, 3, 4)])
    def code(self, request):
        return HollowPlanar3DCode(*request.param)
