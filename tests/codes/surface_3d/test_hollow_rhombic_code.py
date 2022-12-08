import numpy as np
import pytest
from panqec.bpauli import bsf_wt
from panqec.codes import HollowRhombicCode
import panqec.bsparse as bsparse
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestHollowRhombicCode(StabilizerCodeTest):

    @pytest.fixture(params=[4, 6])
    def code(self, request):
        return HollowRhombicCode(request.param)