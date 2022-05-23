import pytest
from panqec.codes import RhombicCode
from tests.tc3d.stabilizer_code_test import StabilizerCodeTest


class TestRhombicCode(StabilizerCodeTest):

    @pytest.fixture
    def code(self):
        return RhombicCode(5, 5, 5)

    def test_stabilizer_index(self, code):
        assert all(
            len(index) in [3, 4] for index in code.stabilizer_index
        )
