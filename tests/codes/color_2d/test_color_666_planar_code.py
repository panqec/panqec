import pytest
from panqec.codes import Color666PlanarCode
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestColor666PlanarCode(StabilizerCodeTest):

    @pytest.fixture(params=[1, 2, 3])
    def code(self, request):
        return Color666PlanarCode(request.param)
