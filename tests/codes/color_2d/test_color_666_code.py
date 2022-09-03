import pytest
from panqec.codes import Color666Code
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestColor666Code(StabilizerCodeTest):

    @pytest.fixture(params=[1, 2, 3])
    def code(self, request):
        return Color666Code(request.param)
