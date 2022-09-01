import pytest
from panqec.codes import Color666Code
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestColor666Code(StabilizerCodeTest):

    @pytest.fixture
    def code(self):
        return Color666Code(7, 7, 7)
