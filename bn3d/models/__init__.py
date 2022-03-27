"""
Toric code in 3D.

:Author:
    Eric Huang
"""
import numpy as np
from ..bpauli import barray_to_bvector, new_barray
from .base._stabilizer_code import StabilizerCode  # noqa
from .surface_2d._toric_2d_code import Toric2DCode  # noqa
from .surface_2d._planar_2d_code import Planar2DCode  # noqa
from .surface_2d._rotated_planar_2d_code import RotatedPlanar2DCode  # noqa
from .surface_3d._toric_3d_code import Toric3DCode  # noqa
from .surface_3d._planar_3d_code import Planar3DCode  # noqa
from .surface_3d._rotated_planar_3d_code import RotatedPlanar3DCode  # noqa
from .rhombic._rhombic_code import RhombicCode  # noqa
from .surface_3d._rotated_toric_3d_code import RotatedToric3DCode  # noqa
from .fractons._xcube_code import XCubeCode  # noqa


def get_vertex_stabilizers(L: int) -> np.ndarray:
    """Z operators on edges around vertices."""
    vertex_stabilizers = []
    for x in range(L):
        for y in range(L):
            for z in range(L):

                # Stabilizer at a vertex
                stabilizer = new_barray(L)

                # Apply Z in x edges
                stabilizer[0, x, y, z, 1] = 1
                stabilizer[0, (x + L - 1) % L, y, z, 1] = 1

                # Apply Z on y edges
                stabilizer[1, x, y, z, 1] = 1
                stabilizer[1, x, (y + L - 1) % L, z, 1] = 1

                # Apply Z on z edges
                stabilizer[2, x, y, z, 1] = 1
                stabilizer[2, x, y, (z + L - 1) % L, 1] = 1

                vertex_stabilizers.append(barray_to_bvector(stabilizer, L))
    return np.array(vertex_stabilizers)


def get_face_stabilizers(L: int) -> np.ndarray:
    """X operators on edges around faces."""
    face_stabilizers = []
    for x in range(L):
        for y in range(L):
            for z in range(L):

                # Apply X in x face
                stabilizer = new_barray(L)
                stabilizer[1, x, y, z, 0] = 1
                stabilizer[2, x, y, z, 0] = 1
                stabilizer[1, x, y, (z + 1) % L, 0] = 1
                stabilizer[2, x, (y + 1) % L, z, 0] = 1
                face_stabilizers.append(barray_to_bvector(stabilizer, L))

                # Apply X in y face
                stabilizer = new_barray(L)
                stabilizer[0, x, y, z, 0] = 1
                stabilizer[2, x, y, z, 0] = 1
                stabilizer[0, x, y, (z + 1) % L, 0] = 1
                stabilizer[2, (x + 1) % L, y, z, 0] = 1
                face_stabilizers.append(barray_to_bvector(stabilizer, L))

                # Apply X in z face
                stabilizer = new_barray(L)
                stabilizer[0, x, y, z, 0] = 1
                stabilizer[1, x, y, z, 0] = 1
                stabilizer[0, x, (y + 1) % L, z, 0] = 1
                stabilizer[1, (x + 1) % L, y, z, 0] = 1
                face_stabilizers.append(barray_to_bvector(stabilizer, L))
    return np.array(face_stabilizers)


def get_all_stabilizers(L):
    face_stabilizers = get_face_stabilizers(L)
    vertex_stabilizers = get_vertex_stabilizers(L)
    stabilizers = np.concatenate([face_stabilizers, vertex_stabilizers])

    return np.array(stabilizers)


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
