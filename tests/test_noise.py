import numpy as np
import pytest
from bn3d.noise import generate_pauli_noise, deform_operator
from bn3d.bpauli import get_bvector_index


class TestGeneratePauliNoise:

    @pytest.fixture(autouse=True)
    def seed_random(self):
        np.random.seed(0)

    def test_generate_pauli_noise(self):
        p_X = 0.1
        p_Y = 0.2
        p_Z = 0.5
        L = 10
        noise = generate_pauli_noise(p_X, p_Y, p_Z, L)
        assert list(noise.shape) == [2*3*L**3]
        assert noise.dtype == np.uint

    def test_no_errors_if_p_zero(self):
        L = 10
        noise = generate_pauli_noise(0, 0, 0, L)
        assert np.all(noise == 0)

    def test_all_X_if_p_X_one(self):
        L = 10
        noise = generate_pauli_noise(1, 0, 0, L)
        assert np.all(noise[:3*L**3] == 1)
        assert np.all(noise[3*L**3:] == 0)

    def test_all_Y_if_p_Y_one(self):
        L = 10
        noise = generate_pauli_noise(0, 1, 0, L)
        assert np.all(noise[:3*L**3] == 1)
        assert np.all(noise[3*L**3:] == 1)

    def test_all_Z_if_p_Z_one(self):
        L = 10
        noise = generate_pauli_noise(0, 0, 1, L)
        assert np.all(noise[:3*L**3] == 0)
        assert np.all(noise[3*L**3:] == 1)

    def test_only_X_if_p_X_only(self):
        L = 10
        noise = generate_pauli_noise(0.5, 0, 0, L)
        assert np.any(noise[:3*L**3] == 1)
        assert np.all(noise[3*L**3:] == 0)

    def test_only_Y_if_p_Y_only(self):
        L = 10
        noise = generate_pauli_noise(0, 0.5, 0, L)
        assert np.all(noise[:3*L**3] == noise[3*L**3:])

    def test_only_Z_if_p_Z_only(self):
        L = 10
        noise = generate_pauli_noise(0, 0, 0.5, L)
        assert np.all(noise[:3*L**3] == 0)
        assert np.any(noise[3*L**3:] == 1)


class TestDeformOperator:

    @pytest.fixture(autouse=True)
    def noise_config(self):
        self.L = 10
        self.noise = generate_pauli_noise(0.1, 0.2, 0.3, self.L)
        self.deformed = deform_operator(self.noise, self.L)

    def test_deform_operator_shape(self):
        assert self.deformed.dtype == np.uint
        assert list(self.deformed.shape) == list(self.noise.shape)

    def test_deformed_is_different(self):
        assert np.any(self.noise != self.deformed)

    def test_deformed_composed_original_has_Ys_only(self):
        composed = (self.deformed + self.noise) % 2
        assert np.all(composed[3*self.L**3:] == composed[:3*self.L**3])

    def test_only_x_edges_are_different(self):
        differing_locations = []
        for edge in range(3):
            for x in range(self.L):
                for y in range(self.L):
                    for z in range(self.L):
                        i_X = get_bvector_index(edge, x, y, z, 0, self.L)
                        i_Z = get_bvector_index(edge, x, y, z, 1, self.L)
                        if (
                            self.deformed[i_X] != self.noise[i_X]
                            or
                            self.deformed[i_Z] != self.noise[i_Z]
                        ):
                            differing_locations.append(
                                (edge, x, y, z)
                            )

        assert len(differing_locations) > 0
        for location in differing_locations:
            edge = location[0]
            assert edge == 0
