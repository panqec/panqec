import pytest
from panqec.codes import RhombicCode
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestRhombicCode(StabilizerCodeTest):

    @pytest.fixture
    def code(self):
        return RhombicCode(6, 6, 6)

    def test_stabilizer_index(self, code):
        assert all(
            len(index) in [3, 4] for index in code.stabilizer_index
        )
