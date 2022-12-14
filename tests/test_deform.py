import pytest
import numpy as np
from panqec.codes import Toric3DCode, StabilizerCode
from panqec.error_models import PauliErrorModel
from panqec.decoders import (
    SweepDecoder3D, SweepMatchDecoder, MatchingDecoder
)


@pytest.fixture
def code():
    return Toric3DCode(3, 4, 5)


@pytest.fixture
def rng():
    np.random.seed(0)
    return np.random


def get_pymatching_distance_matrix(matching):
    edges = matching.edges()
    nodes = sorted(
        set([edge[0] for edge in edges] + [edge[1] for edge in edges])
    )
    weights = {
        (edge[0], edge[1]): edge[2]['weight']
        for edge in edges
    }
    matrix = np.zeros((len(nodes), len(nodes)), dtype=float)
    for i_1, node_1 in enumerate(nodes):
        for i_2, node_2 in enumerate(nodes):
            if (node_1, node_2) in weights:
                matrix[i_1, i_2] = weights[(i_1, i_2)]
    return matrix


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
        error_model = PauliErrorModel(
            *noise, deformation_name='XZZX',
            deformation_kwargs={'deformation_axis': 'z'}
        )
        error = error_model.generate(code, error_rate=1, rng=rng)
        pauli = code.from_bsf(error)
        for edge in code.qubit_coordinates:
            if code.qubit_axis(edge) == 'z':
                assert pauli[edge] == deformed
            else:
                assert pauli[edge] == original

    def test_original_all_X_becomes_Z_on_deformed_axis(self, code):
        error_model = PauliErrorModel(
            1, 0, 0,
            deformation_name='XZZX',
            deformation_kwargs={'deformation_axis': 'z'}
        )
        error = error_model.generate(code, error_rate=1)
        pauli = code.from_bsf(error)
        print(pauli)

        for edge in code.qubit_index:
            if code.qubit_axis(edge) == 'z':
                assert pauli[edge] == 'Z'
            else:
                assert pauli[edge] == 'X'

    def test_original_all_Z_becomes_X_on_deformed_axis(self, code):
        error_model = PauliErrorModel(
            0, 0, 1,
            deformation_name='XZZX',
            deformation_kwargs={'deformation_axis': 'z'}
        )
        error = error_model.generate(code, error_rate=1)
        pauli = code.from_bsf(error)

        for edge in code.qubit_index:
            if code.qubit_axis(edge) == 'z':
                assert pauli[edge] == 'X'
            else:
                assert pauli[edge] == 'Z'

    def test_all_Y_deformed_is_still_all_Y(self, code):
        error_model = PauliErrorModel(
            0, 1, 0,
            deformation_name='XZZX',
            deformation_kwargs={'deformation_axis': 'z'}
        )
        error = error_model.generate(code, error_rate=1)
        pauli = code.from_bsf(error)

        for edge in code.qubit_index:
            assert pauli[edge] == 'Y'

    def test_label(self, code):
        error_model = PauliErrorModel(
            1, 0, 0,
            deformation_name='XZZX',
            deformation_kwargs={'deformation_axis': 'z'}
        )
        assert error_model.label == 'Deformed XZZX Pauli X1.0000Y0.0000Z0.0000'

    def test_decode_trivial(self, code):
        error_model = PauliErrorModel(
            0.1, 0.2, 0.7,
            deformation_name='XZZX',
            deformation_kwargs={'deformation_axis': 'z'}
        )
        error_rate = 0.1
        decoder = SweepMatchDecoder(code, error_model, error_rate)

        syndrome = np.zeros(code.stabilizer_matrix.shape[0], dtype=np.uint)
        correction = decoder.decode(syndrome)
        assert np.all(correction == 0)
        assert issubclass(correction.dtype.type, np.integer)

    def test_decode_single_X_on_undeformed_axis(self, code):
        error_model = PauliErrorModel(
            0.1, 0.2, 0.7,
            deformation_name='XZZX',
            deformation_kwargs={'deformation_axis': 'z'}
        )
        error_rate = 0.1
        decoder = SweepMatchDecoder(code, error_model, error_rate)

        # Single-qubit X error on undeformed edge.
        error = code.to_bsf({
            (0, 1, 0): 'X'
        })
        assert np.any(error != 0)

        # Calculate the syndrome and make sure it's nontrivial.
        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        # Total error should be in code space.
        correction = decoder.decode(syndrome)
        total_error = (error + correction) % 2
        assert np.all(code.measure_syndrome(total_error) == 0)

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
        error_model = PauliErrorModel(
            *noise_direction,
            deformation_name='XZZX',
            deformation_kwargs={'deformation_axis': 'z'}
        )
        error_rate = 0.1
        decoder = SweepMatchDecoder(code, error_model, error_rate)

        # Single-qubit X error on undeformed edge.
        error = code.to_bsf({
            (0, 1, 0): 'X'
        })
        assert np.any(error != 0)

        # Calculate the syndrome and make sure it's nontrivial.
        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        # Total error should be in code space.
        correction = decoder.decode(syndrome)
        total_error = (error + correction) % 2
        assert np.all(code.measure_syndrome(total_error) == 0)

    # TODO fix pymatching
    @pytest.mark.skip
    def test_deformed_pymatching_weights_nonuniform(self, code):
        error_model = PauliErrorModel(
            0.1, 0.2, 0.7,
            deformation_name='XZZX',
            deformation_kwargs={'deformation_axis': 'z'}
        )
        error_rate = 0.1
        decoder = SweepMatchDecoder(code, error_model, error_rate)
        assert decoder.matcher.error_model.direction == (0.1, 0.2, 0.7)
        matching = decoder.matcher.matcher_x
        distance_matrix = get_pymatching_distance_matrix(matching)
        n_vertices = int(np.product(code.size))
        assert distance_matrix.shape == (n_vertices, n_vertices)

        # The index of the origin vertex.
        origin_index = [
            index
            for index, location in enumerate(code.stabilizer_coordinates)
            if location == (0, 0, 0)
            and code.stabilizer_type(location) == 'vertex'
        ][0]

        # Distances from the origin vertex.
        origin_distances = np.zeros(code.size)

        for index, coordinate in enumerate(code.stabilizer_index):
            if code.stabilizer_type(coordinate) == 'vertex':
                location = tuple(
                    (np.array(coordinate)/2).astype(int).tolist()
                )
                origin_distances[location] = distance_matrix[
                    origin_index, index
                ]

        assert origin_distances[0, 0, 0] == 0

        # Distances in the undeformed direction should be equal.
        assert origin_distances[1, 0, 0] == origin_distances[0, 1, 0]

        # Distances in the deformed direction should be different.
        assert origin_distances[0, 1, 0] != origin_distances[0, 0, 1]

    # TODO fix pymatching new version
    @pytest.mark.skip
    def test_equal_XZ_bias_deformed_pymatching_weights_uniform(self, code):
        error_model = PauliErrorModel(
            0.4, 0.2, 0.4,
            deformation_name='XZZX',
            deformation_kwargs={'deformation_axis': 'z'}
        )
        print(f'{error_model.direction=}')
        error_rate = 0.1
        decoder = SweepMatchDecoder(code, error_model, error_rate)
        assert decoder.matcher.error_model.direction == (0.4, 0.2, 0.4)
        matching = decoder.matcher.matcher_x
        distance_matrix = get_pymatching_distance_matrix(matching)
        n_vertices = int(np.product(code.size))
        assert distance_matrix.shape == (n_vertices, n_vertices)

        # Distances from the origin vertex.
        origin_distances = distance_matrix[0].reshape(code.size)
        assert origin_distances[0, 0, 0] == 0

        # Distances in the undeformed direction should be equal.
        assert origin_distances[0, 1, 0] == origin_distances[0, 0, 1]

        # Distances in the deformed direction should be different.
        assert origin_distances[1, 0, 0] == origin_distances[0, 0, 1]


class TestDeformedSweepDecoder3D:

    def test_decode_trivial(self, code):
        error_model = PauliErrorModel(
            1/3, 1/3, 1/3,
            deformation_name='XZZX',
            deformation_kwargs={'deformation_axis': 'z'}
        )
        error_rate = 0.5
        decoder = SweepDecoder3D(code, error_model, error_rate)
        n = code.n
        error = np.zeros(2*n, dtype=np.uint)
        syndrome = code.measure_syndrome(error)
        correction = decoder.decode(syndrome)
        total_error = (correction + error) % 2
        assert np.all(code.measure_syndrome(total_error) == 0)
        assert issubclass(correction.dtype.type, np.integer)

    def test_all_3_faces_active(self, code):
        error_pauli = dict()
        sites = [
            (3, 2, 2), (2, 4, 3)
        ]
        for site in sites:
            error_pauli[site] = 'Z'
        error = code.to_bsf(error_pauli)
        error_model = PauliErrorModel(
            1/3, 1/3, 1/3,
            deformation_name='XZZX',
            deformation_kwargs={'deformation_axis': 'z'}
        )
        error_rate = 0.5
        decoder = SweepDecoder3D(code, error_model, error_rate)
        syndrome = code.measure_syndrome(error)
        correction = decoder.decode(syndrome)
        total_error = (error + correction) % 2
        assert np.all(code.measure_syndrome(total_error) == 0)


class TestMatchingDecoder:

    def test_decode_trivial(self, code):
        error_model = PauliErrorModel(
            1/3, 1/3, 1/3,
            deformation_name='XZZX',
            deformation_kwargs={'deformation_axis': 'z'}
        )
        error_rate = 0.5
        decoder = MatchingDecoder(code, error_model, error_rate,
                                  error_type='X')
        n = code.n
        error = np.zeros(2*n, dtype=np.uint)
        syndrome = code.measure_syndrome(error)
        correction = decoder.decode(syndrome)
        total_error = (correction + error) % 2
        assert np.all(code.measure_syndrome(total_error) == 0)
        assert issubclass(correction.dtype.type, np.integer)


class XNoiseOnYZEdgesOnly(PauliErrorModel):
    """X noise applied on y and z edges only."""

    def __init__(self):
        super(XNoiseOnYZEdgesOnly, self).__init__(1, 0, 0)

    def generate(
        self, code: StabilizerCode, error_rate: float, rng=None
    ) -> np.ndarray:
        error = super(XNoiseOnYZEdgesOnly, self).generate(
            code, error_rate, rng=rng
        )
        for index, location in enumerate(code.qubit_coordinates):
            if code.qubit_axis(location) == 'x':
                error[index] = 0
        return error


class TestMatchingXNoiseOnYZEdgesOnly:

    def test_decode(self, code):
        for seed in range(5):
            rng = np.random.default_rng(seed=seed)
            error_rate = 0.5
            error_model = XNoiseOnYZEdgesOnly()
            decoder = MatchingDecoder(
                code, error_model, error_rate, error_type='X'
            )
            error = error_model.generate(
                code, error_rate=error_rate, rng=rng
            )
            assert any(error), 'Error should be non-trivial'
            syndrome = code.measure_syndrome(error)
            correction = decoder.decode(syndrome)
            assert any(correction), 'Correction should be non-trivial'
            total_error = (correction + error) % 2
            assert np.all(
                code.measure_syndrome(total_error) == 0
            ), 'Total error should be in code space'

            # Error and correction as objects.
            error_pauli = code.from_bsf(error)
            correction_pauli = code.from_bsf(correction)

            x_edges = [
                edge for edge in code.qubit_index
                if code.qubit_axis(edge) == 'x'
            ]
            y_edges = [
                edge for edge in code.qubit_index
                if code.qubit_axis(edge) == 'y'
            ]
            z_edges = [
                edge for edge in code.qubit_index
                if code.qubit_axis(edge) == 'z'
            ]

            assert np.all(
                edge not in error_pauli
                or error_pauli[edge] == 'I'
                for edge in x_edges
            ), 'No errors should be on x edges'

            assert np.all(
                edge not in correction_pauli
                or correction_pauli[edge] == 'I'
                for edge in x_edges
            ), 'No corrections should be on x edges'

            assert np.any([
                correction_pauli[edge] != 'I'
                for edge in y_edges + z_edges
                if edge in correction_pauli
            ]), 'Non-trivial corrections should be on the y and z edges'
