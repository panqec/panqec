import json
import pytest
import numpy as np
from panqec.utils import (
    sizeof_fmt, identity, NumpyEncoder, list_where_str, list_where, set_where,
)


@pytest.mark.parametrize(
    'value, expected_result',
    [
        (0, '0.0B'),
        (1024, '1.0KiB'),
        (1024*1024*1.5, '1.5MiB'),
        (1024**8*1.2, '1.2YiB')
    ]
)
def test_sizeof_fmt(value, expected_result):
    assert sizeof_fmt(value) == expected_result


def test_identity():
    x = [1, 2, 3]
    assert x == identity(x)


def test_numpy_encoder():

    data = {
        'foo': np.array([1, 2, 3], dtype=int),
        'bar': np.array([1.2, 2.3, 3.4], dtype=float),
        'more': {
            'baz': np.array([1], dtype=np.uint)[0],
            'float': np.array([1.5], dtype=np.float32)[0],
            'bar': np.array([1.2, 2.3, 3.4], dtype=float),
        },
        'long': np.array([[3, 2, 1], [1, 2, 3]], dtype=int),
        'hello': (1, 2, 3),
    }
    data_str = json.dumps(data, cls=NumpyEncoder)
    new_data = json.loads(data_str)
    assert new_data == {
        'foo': [1, 2, 3],
        'bar': [1.2, 2.3, 3.4],
        'more': {
            'baz': 1,
            'float': 1.5,
            'bar': [1.2, 2.3, 3.4],
        },
        'long': [[3, 2, 1], [1, 2, 3]],
        'hello': [1, 2, 3],
    }


@pytest.mark.parametrize(
    'array, where_str, where_list, where_set',
    [
        (
            np.array([[0, 1], [1, 0]]),
            '01 10',
            [(0, 1), (1, 0)],
            {(0, 1), (1, 0)},
        ),
        (
            np.array([
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 0, 0],
                ],
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 0, 0],
                ],
                [
                    [0, 0, 1],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
            ]),
            '012 110 202',
            [(0, 1, 2), (1, 1, 0), (2, 0, 2)],
            {(0, 1, 2), (1, 1, 0), (2, 0, 2)},
        )
    ]
)
def test_list_where_str(array, where_str, where_list, where_set):
    assert list_where_str(array) == where_str
    assert list_where(array) == where_list
    assert set_where(array) == where_set
