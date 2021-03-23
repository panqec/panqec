from pymatching import Matching

import numpy as np

def test_getting_started():

    # Five-qubit bit flip code.
    Hz = np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
    ])

    m = Matching(Hz)

    # Noise example IIXXI.
    noise = np.array([0, 0, 1, 1, 0])

    # Syndrome vector.
    z = Hz.dot(noise) % 2
    assert np.all(z == [0, 1, 0, 1])

    # Decode the syndrome.
    c = m.decode(z)

    print(f'C: {c}, of type {type(c)}')
    assert type(c) is np.ndarray

    assert np.all(c == [0, 0, 1, 1, 0])
