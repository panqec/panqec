import pytest
from panqec.codes import Toric2DCode
from panqec.decoders import MemoryBeliefPropagationDecoder
from tests.decoders.decoder_test import DecoderTest


class TestMemoryBeliefPropagationDecoder(DecoderTest):

    @pytest.fixture
    def code(self):
        return Toric2DCode(4)

    @pytest.fixture
    def decoder(self, code, error_model):
        error_rate = 0.1
        return MemoryBeliefPropagationDecoder(code, error_model, error_rate)
