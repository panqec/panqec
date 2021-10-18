"""
Toric code in 3D.

:Author:
    Eric Huang
"""
import numpy as np
from ..bpauli import barray_to_bvector, new_barray
from ._toric_code_3d import ToricCode3D  # noqa
from ._toric_3d_pauli import Toric3DPauli  # noqa
from ._rotated_planar_code_3d import RotatedPlanarCode3D  # noqa
from ._rotated_toric_code_3d import RotatedToricCode3D  # noqa
from ._rotated_planar_3d_pauli import RotatedPlanar3DPauli  # noqa
from ._rotated_toric_3d_pauli import RotatedToric3DPauli  # noqa
from ._pymatching_decoder import Toric3DPymatchingDecoder  # noqa
from ._sweep_decoder_3d import SweepDecoder3D  # noqa
from ._sweep_match_decoder import SweepMatchDecoder  # noqa
from ._rotated_sweep_match_decoder import RotatedSweepMatchDecoder  # noqa


def get_vertex_Z_stabilisers(L: int) -> np.ndarray:
    """Z operators on edges around vertices."""
    vertex_stabilisers = []
    for x in range(L):
        for y in range(L):
            for z in range(L):

                # Stabiliser at a vertex
                stabiliser = new_barray(L)

                # Apply Z in x edges
                stabiliser[0, x, y, z, 1] = 1
                stabiliser[0, (x + L - 1) % L, y, z, 1] = 1

                # Apply Z on y edges
                stabiliser[1, x, y, z, 1] = 1
                stabiliser[1, x, (y + L - 1) % L, z, 1] = 1

                # Apply Z on z edges
                stabiliser[2, x, y, z, 1] = 1
                stabiliser[2, x, y, (z + L - 1) % L, 1] = 1

                vertex_stabilisers.append(barray_to_bvector(stabiliser, L))
    return np.array(vertex_stabilisers)


def get_face_X_stabilisers(L: int) -> np.ndarray:
    """X operators on edges around faces."""
    face_stabilisers = []
    for x in range(L):
        for y in range(L):
            for z in range(L):

                # Apply X in x face
                stabiliser = new_barray(L)
                stabiliser[1, x, y, z, 0] = 1
                stabiliser[2, x, y, z, 0] = 1
                stabiliser[1, x, y, (z + 1) % L, 0] = 1
                stabiliser[2, x, (y + 1) % L, z, 0] = 1
                face_stabilisers.append(barray_to_bvector(stabiliser, L))

                # Apply X in y face
                stabiliser = new_barray(L)
                stabiliser[0, x, y, z, 0] = 1
                stabiliser[2, x, y, z, 0] = 1
                stabiliser[0, x, y, (z + 1) % L, 0] = 1
                stabiliser[2, (x + 1) % L, y, z, 0] = 1
                face_stabilisers.append(barray_to_bvector(stabiliser, L))

                # Apply X in z face
                stabiliser = new_barray(L)
                stabiliser[0, x, y, z, 0] = 1
                stabiliser[1, x, y, z, 0] = 1
                stabiliser[0, x, (y + 1) % L, z, 0] = 1
                stabiliser[1, (x + 1) % L, y, z, 0] = 1
                face_stabilisers.append(barray_to_bvector(stabiliser, L))
    return np.array(face_stabilisers)


def get_all_stabilisers(L):
    face_stabilisers = get_face_X_stabilisers(L)
    vertex_stabilisers = get_vertex_Z_stabilisers(L)
    stabilisers = np.concatenate([face_stabilisers, vertex_stabilisers])

    return np.array(stabilisers)


def get_X_logicals(L):
    """Get the 3 logical X operators."""
    logicals = []

    # X operators along x edges in x direction.
    logical = new_barray(L)
    for x in range(L):
        logical[0, x, 0, 0, 0] = 1
    logicals.append(barray_to_bvector(logical, L))

    # X operators along y edges in y direction.
    logical = new_barray(L)
    for y in range(L):
        logical[1, 0, y, 0, 0] = 1
    logicals.append(barray_to_bvector(logical, L))

    # X operators along z edges in z direction
    logical = new_barray(L)
    for z in range(L):
        logical[2, 0, 0, z, 0] = 1
    logicals.append(barray_to_bvector(logical, L))

    return np.array(logicals)


def get_Z_logicals(L):
    """Get the 3 logical Z operators."""
    logicals = []

    # Z operators on x edges forming surface normal to x (yz plane).
    logical = new_barray(L)
    for y in range(L):
        for z in range(L):
            logical[0, 0, y, z, 1] = 1
    logicals.append(barray_to_bvector(logical, L))

    # Z operators on y edges forming surface normal to y (zx plane).
    logical = new_barray(L)
    for z in range(L):
        for x in range(L):
            logical[1, x, 0, z, 1] = 1
    logicals.append(barray_to_bvector(logical, L))

    # Z operators on z edges forming surface normal to z (xy plane).
    logical = new_barray(L)
    for x in range(L):
        for y in range(L):
            logical[2, x, y, 0, 1] = 1
    logicals.append(barray_to_bvector(logical, L))

    return np.array(logicals)


def get_all_logicals(L):
    X_logicals = get_X_logicals(L)
    Z_logicals = get_Z_logicals(L)
    logicals = np.concatenate([X_logicals, Z_logicals])
    return logicals
