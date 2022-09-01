import pytest
from panqec.codes import Color3DCode
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestColor3DCode(StabilizerCodeTest):

    @pytest.fixture
    def code(self):
        return Color3DCode(4, 4, 4)
