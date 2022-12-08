import pytest
from panqec.codes import RhombicPlanarCode
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestRhombicPlanarCode(StabilizerCodeTest):

    @pytest.fixture
    def code(self):
        return RhombicPlanarCode(4)

    def test_stabilizer_index(self, code):
        assert all(
            len(index) in [3, 4] for index in code.stabilizer_index
        )
