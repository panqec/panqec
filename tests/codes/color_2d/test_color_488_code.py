import pytest
from panqec.codes import Color488Code
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestColor488Code(StabilizerCodeTest):

    @pytest.fixture
    def code(self):
        return Color488Code(5, 5, 5)
