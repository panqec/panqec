import json
import pytest
import numpy as np
from panqec.bsparse import from_array
from panqec.utils import (
    sizeof_fmt, identity, NumpyEncoder, list_where_str, list_where, set_where,
    format_polynomial, simple_print, find_nearest, get_label
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


def test_format_polynomial():
    assert format_polynomial('x', [1, 2, 3], digits=1) == '1.0 + 2.0x + 3.0x^2'
    assert format_polynomial('x', [1.2, 2.1, 3.3], digits=1) == (
        '1.2 + 2.1x + 3.3x^2'
    )
    assert format_polynomial('y', [1.2, -2.1, 3.3], digits=2) == (
        '1.20 - 2.10y + 3.30y^2'
    )
    assert format_polynomial('y', [1.2, 0, 3.3], digits=2) == (
        '1.20 + 3.30y^2'
    )


class TestSimplePrint:

    def test_unint_array(self, capsys):
        a = np.eye(3, dtype=np.uint)
        simple_print(a)
        captured = capsys.readouterr()
        assert captured.out == '100\n010\n001\n'

    def test_sparse_array(self, capsys):
        a = from_array(np.eye(3, dtype=np.uint))
        simple_print(a)
        captured = capsys.readouterr()
        assert captured.out == '100\n010\n001\n'

    def test_sparse_array_space_zeros(self, capsys):
        a = from_array(np.eye(3, dtype=np.uint))
        simple_print(a, zeros=False)
        captured = capsys.readouterr()
        assert captured.out == '1\n 1\n  1\n'

    def test_1d_array(self, capsys):
        a = np.array([1, 0, 1, 0])
        simple_print(a)
        captured = capsys.readouterr()
        assert captured.out == '1010\n'


class TestFindNearest:

    def test_easy_case(self):
        array = [1, 2, 3, 4, 5, 6]
        value = 5.2
        assert find_nearest(array, value) == 5


class TestGetLabel:

    def test_integer_params_only(self):
        label = get_label('Toric2DCode', {'L_x': 5, 'L_y': 6, 'L_z': 7})
        assert label == 'Toric2DCode(L_x=5, L_y=6, L_z=7)'

    def test_float_params_rounded_to_six_digits(self):
        assert get_label('PauliErrorModel', {
            'r_x': 0.33333333333333337,
            'r_y': 0.33333333333333337,
            'r_z': 0.3333333333333333
        }) == 'PauliErrorModel(r_x=0.333333, r_y=0.333333, r_z=0.333333)'
