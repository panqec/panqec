import pytest
from panqec.codes import XCubeCode
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestXCubeCode(StabilizerCodeTest):

    @pytest.fixture
    def code(self):
        return XCubeCode(4)
