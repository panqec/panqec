import itertools
import pytest
import numpy as np
from panqec.bpauli import bcommute
from panqec.codes import Toric3DCode
from panqec.error_models import PauliErrorModel
from panqec.error_models import (
    DeformedXZZXErrorModel
)
from panqec.decoders import (
    DeformedSweepMatchDecoder, DeformedSweepDecoder3D,
    DeformedToric3DPymatchingDecoder, FoliatedMatchingDecoder
)
from panqec.bpauli import bvector_to_pauli_string


@pytest.fixture
def code():
    return Toric3DCode(3, 4, 5)


@pytest.fixture
def rng():
    np.random.seed(0)
    return np.random


@pytest.mark.skip(reason='sparse')
class TestDeformedXZZXErrorModel:

    @pytest.mark.parametrize(
        'noise, original, deformed',
        [
            ((1, 0, 0), 'X', 'Z'),
            ((0, 0, 1), 'Z', 'X'),
            ((0, 1, 0), 'Y', 'Y'),
        ]
    )
    def test_max_noise(self, code, rng, noise, original, deformed):
        error_model = DeformedXZZXErrorModel(*noise)
        error = error_model.generate(code, probability=1, rng=rng)
        pauli = code.from_bsf(error)
        ranges = [range(length) for length in code.shape]
        for edge, x, y, z in itertools.product(*ranges):
            if edge == code.X_AXIS:
                assert pauli.operator((edge, x, y, z)) == deformed
            else:
                assert pauli.operator((edge, x, y, z)) == original

    def test_original_all_X_becomes_Z_on_deformed_axis(self, code):
        error_model = DeformedXZZXErrorModel(1, 0, 0)
        error = error_model.generate(code, probability=1)
        pauli = code.from_bsf(error)

        ranges = [range(length) for length in code.shape]
        for axis, x, y, z in itertools.product(*ranges):
            if axis == 0:
                assert pauli.operator((axis, x, y, z)) == 'Z'
            else:
                assert pauli.operator((axis, x, y, z)) == 'X'

    def test_original_all_Z_becomes_X_on_deformed_axis(self, code):
        error_model = DeformedXZZXErrorModel(0, 0, 1)
        error = error_model.generate(code, probability=1)
        pauli = code.from_bsf(error)

        ranges = [range(length) for length in code.shape]
        for axis, x, y, z in itertools.product(*ranges):
            if axis == 0:
                assert pauli.operator((axis, x, y, z)) == 'X'
            else:
                assert pauli.operator((axis, x, y, z)) == 'Z'

    def test_all_Y_deformed_is_still_all_Y(self, code):
        error_model = DeformedXZZXErrorModel(0, 1, 0)
        error = error_model.generate(code, probability=1)
        pauli = code.from_bsf(error)

        ranges = [range(length) for length in code.shape]
        for axis, x, y, z in itertools.product(*ranges):
            assert pauli.operator((axis, x, y, z)) == 'Y'

    def test_label(self, code):
        error_model = DeformedXZZXErrorModel(1, 0, 0)
        assert error_model.label == 'Deformed XZZX Pauli X1.0000Y0.0000Z0.0000'


# TODO get this to work with Arthur's deform conventions
@pytest.mark.skip(reason='superseded')
class TestDeformOperator:

    @pytest.fixture(autouse=True)
    def undeformed_noise(self, code, rng):
        self.deformed_model = DeformedXZZXErrorModel(0.2, 0.3, 0.5)
        undeformed_model = PauliErrorModel(0.2, 0.3, 0.5)
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
        original_pauli = code.from_bsf(self.noise)
        deformed_pauli = code.from_bsf(self.deformed)

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


@pytest.mark.skip(reason='sparse')
class TestDeformedDecoder:

    def test_decode_trivial(self, code):
        error_model = DeformedXZZXErrorModel(0.1, 0.2, 0.7)
        probability = 0.1
        decoder = DeformedSweepMatchDecoder(error_model, probability)

        syndrome = np.zeros(len(code.stabilizer_matrix), dtype=np.uint)
        correction = decoder.decode(code, syndrome)
        assert np.all(correction == 0)
        assert issubclass(correction.dtype.type, np.integer)

    def test_decode_single_X_on_undeformed_axis(self, code):
        error_model = DeformedXZZXErrorModel(0.1, 0.2, 0.7)
        probability = 0.1
        decoder = DeformedSweepMatchDecoder(error_model, probability)

        # Single-qubit X error on undeformed edge.
        error_pauli = dict()
        error_pauli[(code.Y_AXIS, 0, 0, 0)] = 'X'
        error = code.to_bsf(error_pauli)
        assert np.any(error != 0)

        # Calculate the syndrome and make sure it's nontrivial.
        syndrome = bcommute(code.stabilizer_matrix, error)
        assert np.any(syndrome != 0)

        # Total error should be in code space.
        correction = decoder.decode(code, syndrome)
        total_error = (error + correction) % 2
        assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)

    @pytest.mark.parametrize(
        'operator, location',
        [
            ['X', (0, 0, 0, 0)],
            ['Y', (0, 1, 0, 0)],
            ['Z', (0, 0, 1, 0)],
            ['X', (1, 0, 0, 1)],
            ['Y', (1, 0, 2, 0)],
            ['Z', (1, 2, 0, 0)],
            ['X', (2, 0, 0, 2)],
            ['Y', (2, 1, 1, 0)],
            ['Z', (2, 0, 2, 0)],
        ]
    )
    def test_decode_single_qubit_error(
        self, code, operator, location
    ):
        noise_direction = (0.1, 0.2, 0.7)
        error_model = DeformedXZZXErrorModel(*noise_direction)
        probability = 0.1
        decoder = DeformedSweepMatchDecoder(error_model, probability)

        # Single-qubit X error on undeformed edge.
        error_pauli = dict()
        error_pauli[(code.Y_AXIS, 0, 0, 0)] = 'X'
        error = code.to_bsf(error_pauli)
        assert np.any(error != 0)

        # Calculate the syndrome and make sure it's nontrivial.
        syndrome = bcommute(code.stabilizer_matrix, error)
        assert np.any(syndrome != 0)

        # Total error should be in code space.
        correction = decoder.decode(code, syndrome)
        total_error = (error + correction) % 2
        assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)

    def test_deformed_pymatching_weights_nonuniform(self, code):
        error_model = DeformedXZZXErrorModel(0.1, 0.2, 0.7)
        probability = 0.1
        decoder = DeformedSweepMatchDecoder(error_model, probability)
        assert decoder._matcher._error_model.direction == (0.1, 0.2, 0.7)
        matching = decoder._matcher.get_matcher(code)
        assert matching.stabilizer_graph.distance(0, 0) == 0
        distance_matrix = np.array(matching.stabilizer_graph.all_distances)
        n_vertices = int(np.product(code.size))
        assert distance_matrix.shape == (n_vertices, n_vertices)

        # Distances from the origin vertex.
        origin_distances = distance_matrix[0].reshape(code.size)
        assert origin_distances[0, 0, 0] == 0

        # Distances in the undeformed direction should be equal.
        assert origin_distances[0, 1, 0] == origin_distances[0, 0, 1]

        # Distances in the deformed direction should be different.
        assert origin_distances[1, 0, 0] != origin_distances[0, 0, 1]

    def test_equal_XZ_bias_deformed_pymatching_weights_uniform(self, code):
        error_model = DeformedXZZXErrorModel(0.4, 0.2, 0.4)
        print(f'{error_model.direction=}')
        probability = 0.1
        decoder = DeformedSweepMatchDecoder(error_model, probability)
        assert decoder._matcher._error_model.direction == (0.4, 0.2, 0.4)
        matching = decoder._matcher.get_matcher(code)
        assert matching.stabilizer_graph.distance(0, 0) == 0
        distance_matrix = np.array(matching.stabilizer_graph.all_distances)
        n_vertices = int(np.product(code.size))
        assert distance_matrix.shape == (n_vertices, n_vertices)

        # Distances from the origin vertex.
        origin_distances = distance_matrix[0].reshape(code.size)
        assert origin_distances[0, 0, 0] == 0

        # Distances in the undeformed direction should be equal.
        assert origin_distances[0, 1, 0] == origin_distances[0, 0, 1]

        # Distances in the deformed direction should be different.
        assert origin_distances[1, 0, 0] == origin_distances[0, 0, 1]


@pytest.mark.skip(reason='sparse')
class TestDeformedSweepDecoder3D:

    @pytest.mark.parametrize(
        'noise_direction, expected_edge_probs',
        [
            [(0.9, 0, 0.1), (0.81818, 0.090909, 0.090909)],
            [(0.1, 0, 0.9), (0.05263, 0.47368, 0.47368)],
            [(1/3, 1/3, 1/3), (0.3333, 0.3333, 0.3333)],
            [(0, 0, 1), (0, 0.5, 0.5)],
        ]
    )
    def test_get_edge_probabilities(
        self, code, noise_direction, expected_edge_probs
    ):
        error_model = DeformedXZZXErrorModel(*noise_direction)
        probability = 0.5
        decoder = DeformedSweepDecoder3D(error_model, probability)
        print(decoder.get_edge_probabilities())
        np.testing.assert_allclose(
            decoder.get_edge_probabilities(),
            expected_edge_probs,
            rtol=0.01
        )

    def test_decode_trivial(self, code):
        error_model = DeformedXZZXErrorModel(1/3, 1/3, 1/3)
        probability = 0.5
        decoder = DeformedSweepDecoder3D(error_model, probability)
        n = code.n
        error = np.zeros(2*n, dtype=np.uint)
        syndrome = bcommute(code.stabilizer_matrix, error)
        correction = decoder.decode(code, syndrome)
        total_error = (correction + error) % 2
        assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)
        assert issubclass(correction.dtype.type, np.integer)

    def test_all_3_faces_active(self, code):
        error_pauli = dict()
        sites = [
            (0, 1, 1, 1), (2, 1, 2, 1)
        ]
        for site in sites:
            error_pauli[site] = 'Z'
        error = code.to_bsf(error_pauli)
        error_model = DeformedXZZXErrorModel(1/3, 1/3, 1/3)
        probability = 0.5
        decoder = DeformedSweepDecoder3D(error_model, probability)
        syndrome = bcommute(code.stabilizer_matrix, error)
        correction = decoder.decode(code, syndrome)
        total_error = (error + correction) % 2
        assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)


@pytest.mark.skip(reason='sparse')
class TestDeformedToric3DPymatchingDecoder:

    def test_decode_trivial(self, code):
        error_model = DeformedXZZXErrorModel(1/3, 1/3, 1/3)
        probability = 0.5
        decoder = DeformedToric3DPymatchingDecoder(error_model, probability)
        n = code.n
        error = np.zeros(2*n, dtype=np.uint)
        syndrome = bcommute(code.stabilizer_matrix, error)
        correction = decoder.decode(code, syndrome)
        total_error = (correction + error) % 2
        assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)
        assert issubclass(correction.dtype.type, np.integer)


@pytest.mark.skip(reason='sparse')
class TestMatchingXNoiseOnYZEdgesOnly:

    def test_decode(self, code):
        for seed in range(5):
            rng = np.random.default_rng(seed=seed)
            probability = 0.5
            error_model = XNoiseOnYZEdgesOnly()
            decoder = DeformedToric3DPymatchingDecoder(
                error_model, probability
            )
            error = error_model.generate(
                code, probability=probability, rng=rng
            )
            assert any(error), 'Error should be non-trivial'
            syndrome = bcommute(code.stabilizer_matrix, error)
            correction = decoder.decode(code, syndrome)
            assert any(correction), 'Correction should be non-trivial'
            total_error = (correction + error) % 2
            assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0), (
                'Total error should be in code space'
            )

            # Error and correction as objects.
            error_pauli = code.from_bsf(error)
            correction_pauli = code.from_bsf(correction)

            # Lattice vertex locations.
            locations = list(itertools.product(*[
                range(length) for length in code.size
            ]))

            assert all(
                error_pauli.operator((0, x, y, z)) == 'I'
                for x, y, z in locations
            ), 'No errors should be on x edges'

            assert all(
                correction_pauli.operator((0, x, y, z)) == 'I'
                for x, y, z in locations
            ), 'No corrections should be on x edges'

            assert any([
                correction_pauli.operator((1, x, y, z)) != 'I'
                or correction_pauli.operator((2, x, y, z)) != 'I'
                for x, y, z in locations
            ]), 'Non-trivial corrections should be on the y and z edges'


@pytest.mark.skip(reason='sparse')
class TestFoliatedDecoderXNoiseOnYZEdgesOnly:

    def test_decode(self, code):
        for seed in range(5):
            rng = np.random.default_rng(seed=seed)
            probability = 0.5
            error_model = XNoiseOnYZEdgesOnly()
            decoder = FoliatedMatchingDecoder()
            error = error_model.generate(
                code, probability=probability, rng=rng
            )
            assert any(error), 'Error should be non-trivial'
            syndrome = bcommute(code.stabilizer_matrix, error)
            correction = decoder.decode(code, syndrome)
            assert any(correction), 'Correction should be non-trivial'
            total_error = (correction + error) % 2
            assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0), (
                'Total error should be in code space'
            )

            # Error and correction as objects.
            error_pauli = code.from_bsf(error)
            correction_pauli = code.from_bsf(correction)

            # Lattice vertex locations.
            locations = list(itertools.product(*[
                range(length) for length in code.size
            ]))

            assert all(
                error_pauli.operator((0, x, y, z)) == 'I'
                for x, y, z in locations
            ), 'No errors should be on x edges'

            assert all(
                correction_pauli.operator((0, x, y, z)) == 'I'
                for x, y, z in locations
            ), 'No corrections should be on x edges'

            assert any([
                correction_pauli.operator((1, x, y, z)) != 'I'
                or correction_pauli.operator((2, x, y, z)) != 'I'
                for x, y, z in locations
            ]), 'Non-trivial corrections should be on the y and z edges'
