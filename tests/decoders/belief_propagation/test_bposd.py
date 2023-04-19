import pytest
from panqec.codes import Toric3DCode
from panqec.decoders import BeliefPropagationOSDDecoder
from tests.decoders.decoder_test import DecoderTest


class TestBeliefPropagationOSDDecoder(DecoderTest):

    @pytest.fixture
    def code(self):
        return Toric3DCode(4)

    @pytest.fixture
    def decoder(self, code, error_model):
        error_rate = 0.1
        return BeliefPropagationOSDDecoder(code, error_model, error_rate)
