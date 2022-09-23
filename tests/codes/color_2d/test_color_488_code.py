import pytest
from panqec.codes import Color488Code
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestColor488Code(StabilizerCodeTest):

    @pytest.fixture(params=[1, 2, 3])
    def code(self, request):
        return Color488Code(request.param)
