import pytest
from panqec.codes import XCubeCode
from panqec.decoders import XCubeMatchingDecoder
from tests.decoders.decoder_test import DecoderTest


class TestXCubeMatchingDecoder(DecoderTest):
    @pytest.fixture
    def code(self):
        return XCubeCode(4)

    @pytest.fixture
    def decoder(self, code, error_model):
        error_rate = 0.1
        return XCubeMatchingDecoder(code, error_model, error_rate)
