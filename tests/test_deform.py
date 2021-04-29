import itertools
import pytest
import numpy as np
from bn3d.tc3d import ToricCode3D, Toric3DPauli
from bn3d.deform import DeformedPauliErrorModel
from bn3d.bpauli import bvector_to_pauli_string


@pytest.fixture
def code():
    return ToricCode3D(3, 4, 5)


@pytest.fixture
def rng():
    np.random.seed(0)
    return np.random


class TestDeformedPauliErrorModel:

    @pytest.mark.parametrize(
        'noise, original, deformed',
        [
            ((1, 0, 0), 'X', 'Z'),
            ((0, 0, 1), 'Z', 'X'),
            ((0, 1, 0), 'Y', 'Y'),
        ]
    )
    def test_max_noise(self, code, rng, noise, original, deformed):
        error_model = DeformedPauliErrorModel(*noise)
        error = error_model.generate(code, probability=1, rng=rng)
        pauli = Toric3DPauli(code, bsf=error)
        ranges = [range(length) for length in code.shape]
        for edge, x, y, z in itertools.product(*ranges):
            if edge == code.X_AXIS:
                assert pauli.operator((edge, x, y, z)) == deformed
            else:
                assert pauli.operator((edge, x, y, z)) == original


class TestDeformOperator:

    @pytest.fixture(autouse=True)
    def undeformed_noise(self, code, rng):
        self.deformed_model = DeformedPauliErrorModel(0.2, 0.3, 0.5)
        undeformed_model = self.deformed_model._undeformed_model
        probability = 1
        self.noise = undeformed_model.generate(
            code, probability, rng=rng
        ).copy()
        self.deformed = self.deformed_model._deform_operator(code, self.noise)

    def test_deform_again_gives_original(self, code):
        self.deformed_again = self.deformed_model._deform_operator(
            code, self.deformed
        )
        assert np.all(self.deformed_again == self.noise)

    def test_deform_operator_shape(self):
        assert list(self.deformed.shape) == list(self.noise.shape)

    def test_deformed_is_different(self):
        assert np.any(self.noise != self.deformed)

    def test_deformed_composed_original_has_Ys_only(self, code):
        L_x, L_y, L_z = code.size
        composed = (self.deformed + self.noise) % 2
        set(list(bvector_to_pauli_string(composed))) == set(['I', 'Y'])

    def test_only_x_edges_are_different(self, code):
        L_x, L_y, L_z = code.size
        original_pauli = Toric3DPauli(code, bsf=self.noise)
        deformed_pauli = Toric3DPauli(code, bsf=self.deformed)

        ranges = [range(length) for length in code.shape]

        differing_locations = []
        differing_operators = []
        for edge, x, y, z in itertools.product(*ranges):
            original_operator = original_pauli.operator((edge, x, y, z))
            deformed_operator = deformed_pauli.operator((edge, x, y, z))
            if original_operator != deformed_operator:
                differing_locations.append((edge, x, y, z))
                differing_operators.append(
                    (original_operator, deformed_operator)
                )

        assert len(differing_locations) > 0

        differing_edges = [location[0] for location in differing_locations]
        assert all([edge == 0 for edge in differing_edges])
