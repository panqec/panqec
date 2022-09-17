import os
import json
from typing import Tuple, List
import pytest
import numpy as np
from panqec.codes import RotatedToric3DCode
from panqec.bpauli import apply_deformation
from panqec.error_models import DeformedXZZXErrorModel
from panqec.decoders import BeliefPropagationOSDDecoder

from tests.codes.stabilizer_code_test import StabilizerCodeTestWithCoordinates


class RotatedToric3DCodeTest(StabilizerCodeTestWithCoordinates):
    code_class = RotatedToric3DCode


class TestRotatedToric3DCode2x2x2(RotatedToric3DCodeTest):
    size = (2, 2, 2)
    expected_plane_edges_xy = [
        (1, 1), (3, 1),
        (1, 3), (3, 3),
    ]
    expected_vertical_faces_xy = [
        (1, 1), (1, 3),
        (3, 1), (3, 3),
    ]
    expected_plane_faces_xy = [
        (2, 2), (4, 4)
    ]
    expected_plane_vertices_xy = [
        (4, 2), (2, 4)
    ]
    expected_plane_z = [1, 3]
    expected_vertical_z = [2]


class TestRotatedToric3DCode3x2x2(RotatedToric3DCodeTest):
    size = (3, 2, 2)
    expected_plane_edges_xy = [
        (1, 1), (3, 1), (5, 1),
        (1, 3), (3, 3), (5, 3),
    ]
    expected_vertical_faces_xy = [
        (3, 1), (3, 3),
        (5, 1), (5, 3)
    ]
    expected_plane_faces_xy = [
        (2, 2), (4, 4), (6, 2),
    ]
    expected_plane_vertices_xy = [
        (4, 2), (2, 4), (6, 4)
    ]
    expected_plane_z = [1, 3]
    expected_vertical_z = [2]


@pytest.mark.skip(reason='odd by odd')
class TestRotatedToric3DCode3x3x3(RotatedToric3DCodeTest):
    size = (3, 3, 3)
    expected_plane_edges_xy = [
        (1, 1), (3, 1), (5, 1),
        (1, 3), (3, 3), (5, 3),
        (1, 5), (3, 5), (5, 5),
    ]
    expected_vertical_faces_xy = [
        (3, 3), (3, 5),
        (5, 3), (5, 5),
    ]
    expected_plane_faces_xy = [
        (2, 2), (6, 2),
        (4, 4),
        (2, 6), (6, 6),
    ]
    expected_plane_vertices_xy = [
        (4, 2),
        (2, 4), (6, 4),
        (4, 6),
    ]
    expected_plane_z = [1, 3, 5, 7]
    expected_vertical_z = [2, 4, 6]


class TestRotatedToric3DCode4x3x3(RotatedToric3DCodeTest):
    size = (4, 3, 3)
    expected_plane_edges_xy = [
        (1, 1), (3, 1), (5, 1), (7, 1),
        (1, 3), (3, 3), (5, 3), (7, 3),
        (1, 5), (3, 5), (5, 5), (7, 5),
    ]
    expected_vertical_faces_xy = [
        (1, 3), (1, 5),
        (3, 3), (3, 5),
        (5, 3), (5, 5),
        (7, 3), (7, 5),
    ]
    expected_plane_faces_xy = [
        (2, 2), (6, 2),
        (4, 4), (8, 4),
        (2, 6), (6, 6),
    ]
    expected_plane_vertices_xy = [
        (4, 2), (8, 2),
        (2, 4), (6, 4),
        (4, 6), (8, 6),
    ]
    expected_plane_z = [1, 3, 5]
    expected_vertical_z = [2, 4]


class TestRotatedToric3DCode3x4x3(RotatedToric3DCodeTest):
    size = (3, 4, 3)
    expected_plane_edges_xy = [
        (1, 1), (1, 3), (1, 5), (1, 7),
        (3, 1), (3, 3), (3, 5), (3, 7),
        (5, 1), (5, 3), (5, 5), (5, 7),
    ]
    expected_vertical_faces_xy = [
        (3, 1), (3, 3), (3, 5), (3, 7),
        (5, 1), (5, 3), (5, 5), (5, 7),
    ]
    expected_plane_faces_xy = [
        (2, 2), (2, 6),
        (4, 4), (4, 8),
        (6, 2), (6, 6),
    ]
    expected_plane_vertices_xy = [
        (2, 4), (2, 8),
        (4, 2), (4, 6),
        (6, 4), (6, 8),
    ]
    expected_plane_z = [1, 3, 5]
    expected_vertical_z = [2, 4]


class TestRotatedToric3DCode4x4x4(RotatedToric3DCodeTest):
    size = (4, 4, 4)
    expected_plane_edges_xy = [
        (1, 1), (1, 3), (1, 5), (1, 7),
        (3, 1), (3, 3), (3, 5), (3, 7),
        (5, 1), (5, 3), (5, 5), (5, 7),
        (7, 1), (7, 3), (7, 5), (7, 7),
    ]
    expected_vertical_faces_xy = [
        (1, 1), (1, 3), (1, 5), (1, 7),
        (3, 1), (3, 3), (3, 5), (3, 7),
        (5, 1), (5, 3), (5, 5), (5, 7),
        (7, 1), (7, 3), (7, 5), (7, 7),
    ]
    expected_plane_faces_xy = [
        (2, 2), (2, 6),
        (4, 4), (4, 8),
        (6, 2), (6, 6),
        (8, 4), (8, 8),
    ]
    expected_plane_vertices_xy = [
        (2, 4), (2, 8),
        (4, 2), (4, 6),
        (6, 4), (6, 8),
        (8, 2), (8, 6),
    ]
    expected_plane_z = [1, 3, 5, 7]
    expected_vertical_z = [2, 4, 6]


class TestRotatedToric3DDeformation:

    def test_deformation_index(self):
        code = RotatedToric3DCode(3, 4, 4)
        error_model = DeformedXZZXErrorModel(0.2, 0.3, 0.5)
        deformation_index = error_model.get_deformation_indices(code)
        coords_map = {
            index: coord for index, coord in enumerate(code.qubit_coordinates)
        }
        coords = [coords_map[index] for index in range(len(coords_map))]
        deformation_sites = sorted(
            [
                coords[i]
                for i, active in enumerate(deformation_index)
                if active
            ],
            key=lambda x: x[::-1]
        )
        assert all(z % 2 == 1 for x, y, z in deformation_sites)
        expected_sites_plane = [
            (1, 1), (5, 1), (3, 3), (1, 5), (5, 5), (3, 7)
        ]
        expected_sites = []
        for z in [1, 3, 5, 7]:
            expected_sites += [(x, y, z) for x, y in expected_sites_plane]
        expected_sites.sort(key=lambda x: x[::-1])
        assert deformation_sites == expected_sites

    @pytest.mark.skip(reason='analytics')
    def test_stabilizer_matrix(self):
        """Run this to generate stabilizer matrix for SageMath.

        The data will be saved as json files in the temp directory.
        The data can then be used as input for a SageMath notebook
        to find the lowest-weight Z-only logical operators by brute force.
        """
        project_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(__file__))
        )
        entries = []
        for size in [3, 4, 5, 6, 7, 8, 9, 10]:
            L_x, L_y, L_z = size, size + 1, size
            print(L_x, L_y, L_z)
            code = RotatedToric3DCode(L_x, L_y, L_z)
            out_json = os.path.join(
                project_dir, 'temp', 'rotated_toric_3d_test.json'
                )
            error_model = DeformedXZZXErrorModel(0.2, 0.3, 0.5)
            deformation_index = error_model._get_deformation_indices(code)

            entries.append({
                'size': [L_x, L_y, L_z],
                'coords': code.qubit_coordinates,
                'undeformed': {
                    'n': code.n,
                    'k': code.k,
                    'd': code.d,
                    'stabilizers': code.stabilizer_matrix.tolist(),
                    'logicals_x': code.logicals_x.tolist(),
                    'logicals_z': code.logicals_z.tolist(),
                },
                'deformed': {
                    'n': code.n,
                    'k': code.k,
                    'd': code.d,
                    'stabilizers': apply_deformation(
                        deformation_index, code.stabilizer_matrix
                    ).tolist(),
                    'logicals_x': apply_deformation(
                        deformation_index, code.logicals_x
                    ).tolist(),
                    'logicals_z': apply_deformation(
                        deformation_index, code.logicals_z
                    ).tolist(),
                }
            })
        with open(out_json, 'w') as f:
            json.dump({
                'entries': entries
            }, f)


class TestBPOSDOnRotatedToric3DCodeOddTimesEven:

    @pytest.mark.parametrize('pauli', ['X', 'Y', 'Z'])
    def test_decode_single_qubit_error_bposd(self, pauli):
        code = RotatedToric3DCode(3, 4, 3)
        error_model = DeformedXZZXErrorModel(1/3, 1/3, 1/3)
        error_rate = 0.1
        decoder = BeliefPropagationOSDDecoder(code, error_model, error_rate)

        failing_cases: List[Tuple[int, int, int]] = []
        for site in code.qubit_coordinates:
            error_pauli = dict()
            error_pauli[site] = pauli
            error = code.to_bsf(error_pauli)
            syndrome = code.measure_syndrome(error)
            correction = decoder.decode(syndrome)
            total_error = (error + correction) % 2
            if not np.all(code.measure_syndrome(total_error) == 0):
                failing_cases.append(site)
        n_failing = len(failing_cases)
        max_show = 100
        if n_failing > max_show:
            failing_cases_show = ', '.join(map(str, failing_cases[:max_show]))
            end_part = f'...{n_failing - max_show} more'
        else:
            failing_cases_show = ', '.join(map(str, failing_cases))
            end_part = ''

        assert n_failing == 0, (
            f'Failed decoding {n_failing} {pauli} errors at '
            f'{failing_cases_show} {end_part}'
        )
