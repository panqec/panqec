import numpy as np
from .bpauli import barray_to_bvector, new_barray


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
