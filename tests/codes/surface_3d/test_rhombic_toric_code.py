import pytest
from panqec.codes import RhombicToricCode
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestRhombicToricCode(StabilizerCodeTest):

    @pytest.fixture
    def code(self):
        return RhombicToricCode(6, 6, 6)

    def test_stabilizer_index(self, code):
        assert all(
            len(index) in [3, 4] for index in code.stabilizer_index
        )
