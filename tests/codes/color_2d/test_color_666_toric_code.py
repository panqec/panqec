import pytest
from panqec.codes import Color666ToricCode
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestColor666ToricCode(StabilizerCodeTest):

    @pytest.fixture(params=[1, 2, 3])
    def code(self, request):
        return Color666ToricCode(request.param)
