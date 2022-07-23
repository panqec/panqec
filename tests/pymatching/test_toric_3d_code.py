"""
Test 3D toric code with Pymatching.

:Author:
    Eric Huang
"""

import numpy as np
from panqec.codes import Toric3DCode, Toric2DCode
from panqec.error_models import PauliErrorModel
from panqec.decoders import SweepMatchDecoder
from pymatching import Matching


def test_matching_2x2x2_toric_code():
    rng = np.random.default_rng(seed=0)
    code = Toric3DCode(2, 2, 2)
    error_model = PauliErrorModel(1, 0, 0)
    error_rate = 0.0
    error = error_model.generate(code, error_rate, rng=rng)
    syndrome = code.measure_syndrome(error)
    decoder = SweepMatchDecoder(code, error_model, error_rate)
    correction = decoder.decode(syndrome)
    total_error = (correction + error) % 2
    assert np.all(total_error == 0)


def test_matching_2x2_toric_code():
    rng = np.random.default_rng(seed=0)
    code = Toric2DCode(2, 2, 2)
    error_model = PauliErrorModel(1, 0, 0)
    error_rate = 0.0
    error = error_model.generate(code, error_rate, rng=rng)
    syndrome = code.measure_syndrome(error)
    decoder = SweepMatchDecoder(code, error_model, error_rate)
    correction = decoder.decode(syndrome)
    total_error = (correction + error) % 2
    assert np.all(total_error == 0)


def test_pymatching_minimalist_2d_problem():
    H = np.array([
        [1, 0, 1, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 0, 1, 1]
    ], dtype=np.uint)
    syndrome = np.array([
        0, 0, 0, 0
    ], dtype=np.uint)
    matcher = Matching(H)
    matcher.decode(syndrome)


def test_pymatching_minimalist_3d_problem():
    H = np.array([[
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0
    ], [
        0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0
    ], [
        0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0
    ], [
        0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0
    ], [
        1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0
    ], [
        0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0
    ], [
        0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1
    ], [
        0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1
    ]])
    syndrome = [
        0, 0, 0, 0, 0, 0, 0, 0
    ]
    matcher = Matching(H)
    matcher.decode(syndrome)
