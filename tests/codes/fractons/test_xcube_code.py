import pytest
from panqec.codes import XCubeCode
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestXCubeCode(StabilizerCodeTest):

    @pytest.fixture(params=[2, 3])
    def code(self, request):
        return XCubeCode(request.param)
