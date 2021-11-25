import pytest
from bn3d.tc3d import LayeredRotatedToricCode
from .indexed_code_test import IndexedCodeTest


class TestLayeredRotatedToricCode(IndexedCodeTest):

    L_x = 3
    L_y = 4
    L_z = 3

    @pytest.fixture
    def code(self):
        new_code = LayeredRotatedToricCode(self.L_x, self.L_y, self.L_z)
        return new_code


class TestLayeredRotatedToricPauli:

    L_x = 3
    L_y = 4
    L_z = 3

    @pytest.fixture
    def code(self):
        """Example code with co-prime x and y dimensions."""
        new_code = LayeredRotatedToricCode(self.L_x, self.L_y, self.L_z)
        return new_code
