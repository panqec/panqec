import pytest
import numpy as np
from bn3d.statmech.rbim2d import RandomBondIsingModel2D, WilsonLoop2D


class TestObservables:
    L_x = 5
    L_y = 6

    @pytest.fixture(autouse=True)
    def model(self):
        """An instance of a model with default."""
        model = RandomBondIsingModel2D(self.L_x, self.L_y)
        model.rng = np.random.default_rng(seed=0)
        model.init_spins(np.ones_like(model.spins))
        model.init_disorder(np.ones_like(model.disorder))
        return model

    def test_wilson_loops(self):
        model1 = RandomBondIsingModel2D(5, 6)

        model1.init_spins(np.array([[-1,  1, -1,  1,  1,  1],
                                    [ 1, -1,  1,  1,  1,  1],  # noqa
                                    [-1,  1, -1,  1,  1,  1],
                                    [ 1,  1,  1,  1,  1,  1],  # noqa
                                    [ 1,  1,  1,  1,  1,  1]]))  # noqa
        wl = WilsonLoop2D(model1)
        value_array = wl.evaluate(model1)

        assert value_array.shape == (3,)
        assert np.all(value_array == np.array([-1, 1, -1]))
