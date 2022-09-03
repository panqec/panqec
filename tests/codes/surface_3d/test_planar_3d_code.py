import pytest
from panqec.codes import Planar3DCode
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestPlanar3DCode(StabilizerCodeTest):

    @pytest.fixture
    def code(self):
        return Planar3DCode(4)
