import pytest
from panqec.codes import Color3DCode
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestColor3DCode(StabilizerCodeTest):

    @pytest.fixture(params=[2, 4])
    def code(self, request):
        return Color3DCode(request.param)
