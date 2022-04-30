import numpy as np
from panqec.statmech.loop2d import LoopModel2D, WilsonLoop2D


class TestObservables:
    L_x = 5
    L_y = 6

    def test_wilson_loops(self):
        model = np.empty((6,), dtype="O")
        model[0] = LoopModel2D(2, 2)
        model[1] = LoopModel2D(2, 2)
        model[2] = LoopModel2D(2, 2)
        model[3] = LoopModel2D(2, 2)
        model[4] = LoopModel2D(5, 5)
        model[5] = LoopModel2D(5, 5)

        model[0].init_spins(np.array([[ 0, -1,  0,  1],  # noqa
                                      [ 1,  0, -1,  0],  # noqa
                                      [ 0,  1,  0,  1],  # noqa
                                      [ 1,  0,  1,  0]]))  # noqa
        model[1].init_spins(np.array([[ 0, -1,  0,  1],  # noqa
                                      [-1,  0, -1,  0],  # noqa
                                      [ 0, -1,  0,  1],  # noqa
                                      [ 1,  0,  1,  0]]))  # noqa
        model[2].init_spins(np.array([[ 0, -1,  0,  1],  # noqa
                                      [-1,  0, -1,  0],  # noqa
                                      [ 0, -1,  0,  1],  # noqa
                                      [ 1,  0, -1,  0]]))  # noqa
        model[3].init_spins(np.array([[ 0, -1,  0,  1],  # noqa
                                      [-1,  0, -1,  0],  # noqa
                                      [ 0, -1,  0, -1],  # noqa
                                      [ 1,  0, -1,  0]]))  # noqa
        model[4].init_spins(np.array([[ 0,  1,  0, -1,  0,  1,  0,  1,  0,  1],  # noqa
                                      [-1,  0,  1,  0, -1,  0,  1,  0,  1,  0],  # noqa
                                      [ 0, -1,  0,  1,  0,  1,  0,  1,  0,  1],  # noqa
                                      [-1,  0,  1,  0,  1,  0, -1,  0,  1,  0],  # noqa
                                      [ 0, -1,  0,  1,  0, -1,  0, -1,  0,  1],  # noqa
                                      [ 1,  0,  1,  0, -1,  0,  1,  0,  1,  0],  # noqa
                                      [ 0, -1,  0, -1,  0, -1,  0, -1,  0,  1],  # noqa
                                      [ 1,  0,  1,  0, -1,  0,  1,  0,  1,  0],  # noqa
                                      [ 0,  1,  0,  1,  0,  1,  0,  1,  0,  1],  # noqa
                                      [ 1,  0,  1,  0,  1,  0,  1,  0,  1,  0]]))  # noqa
        model[5].init_spins(np.array([[ 0,  1,  0, -1,  0,  1,  0,  1,  0,  1],  # noqa
                                      [-1,  0, -1,  0, -1,  0, -1,  0,  1,  0],  # noqa
                                      [ 0, -1,  0,  1,  0, -1,  0, -1,  0,  1],  # noqa
                                      [-1,  0,  1,  0, -1,  0,  1,  0,  1,  0],  # noqa
                                      [ 0, -1,  0,  1,  0,  1,  0, -1,  0,  1],  # noqa
                                      [ 1,  0,  1,  0, -1,  0,  1,  0,  1,  0],  # noqa
                                      [ 0, -1,  0,  1,  0, -1,  0, -1,  0,  1],  # noqa
                                      [ 1,  0, -1,  0, -1,  0, -1,  0,  1,  0],  # noqa
                                      [ 0,  1,  0,  1,  0,  1,  0,  1,  0,  1],  # noqa
                                      [ 1,  0,  1,  0,  1,  0,  1,  0,  1,  0]]))  # noqa

        value_array = []
        for i in range(len(model)):
            wl = WilsonLoop2D(model[i])
            value_array.append(wl.evaluate(model[i]))

        print(value_array)
        assert np.all(value_array[0] == np.array([-1]))
        assert np.all(value_array[1] == np.array([1]))
        assert np.all(value_array[2] == np.array([-1]))
        assert np.all(value_array[3] == np.array([1]))
        assert np.all(value_array[4] == np.array([-1, -1, -1]))
        assert np.all(value_array[5] == np.array([1, 1, 1]))
