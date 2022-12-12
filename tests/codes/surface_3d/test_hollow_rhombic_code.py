import pytest
from panqec.codes import HollowRhombicCode
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestHollowRhombicCode(StabilizerCodeTest):

    @pytest.fixture(params=[4, 6])
    def code(self, request):
        return HollowRhombicCode(request.param)
