import pytest
import numpy as np
from panqec.statmech.rbim2d import RandomBondIsingModel2D, WilsonLoop2D, Energy


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
        model = np.empty((5,), dtype="O")
        model[0] = RandomBondIsingModel2D(5, 6)
        model[1] = RandomBondIsingModel2D(8, 8)
        model[2] = RandomBondIsingModel2D(1, 1)
        model[3] = RandomBondIsingModel2D(1, 1)
        model[4] = RandomBondIsingModel2D(2, 2)

        model[0].init_spins(np.array([[-1,  1, -1,  1,  1,  1],
                                      [ 1, -1,  1,  1,  1,  1],  # noqa
                                      [-1,  1, -1,  1,  1,  1],
                                      [ 1,  1,  1,  1,  1,  1],  # noqa
                                      [ 1,  1,  1,  1,  1,  1]]))  # noqa

        model[1].init_spins(-np.ones((8, 8)))
        model[2].init_spins(np.array([[1]]))
        model[3].init_spins(np.array([[-1]]))
        model[4].init_spins(np.array([[-1, 1],
                                      [ 1, -1]]))  # noqa

        value_array = []
        for i in range(len(model)):
            wl = WilsonLoop2D(model[i])
            value_array.append(wl.evaluate(model[i]))

        assert np.all(value_array[0] == np.array([-1, 1, 1]))
        assert np.all(value_array[1] == np.array([-1, 1, 1, 1]))
        assert np.all(value_array[2] == np.array([1]))
        assert np.all(value_array[3] == np.array([-1]))
        assert np.all(value_array[4] == np.array([-1]))

    def test_energy(self):
        model = RandomBondIsingModel2D(5, 7)
        model.init_spins()
        assert Energy().evaluate(model) == model.total_energy()
