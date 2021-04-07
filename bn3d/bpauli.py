"""
Bit array or vector representations of Paulis.
"""
import numpy as np


def barray_to_bvector(a: np.ndarray, L: int) -> np.ndarray:
    """Convert shape (3, L, L, L, 2) binary array to binary vector."""
    return np.concatenate([
        a[:, :, :, :, 0].transpose((1, 2, 3, 0)).reshape(3*L**3),
        a[:, :, :, :, 1].transpose((1, 2, 3, 0)).reshape(3*L**3)
    ])


def get_bvector_index(
    edge: int, x: int, y: int, z: int, block: int, L: int
) -> int:
    return block*3*L**3 + x*3*L**2 + y*3*L + z*3 + edge


def bvector_to_barray(s: np.ndarray, L: int) -> np.ndarray:
    """Convert binary vector to shape (3, L, L, L, 2) binary array."""
    a = np.zeros((3, L, L, L, 2), dtype=s.dtype)
    for edge in range(3):
        for x in range(L):
            for y in range(L):
                for z in range(L):
                    for block in range(2):
                        index = get_bvector_index(edge, x, y, z, block, L)
                        a[edge, x, y, z, block] = s[index]
    return a


def new_barray(L: int) -> np.ndarray:
    return np.zeros((3, L, L, L, 2), dtype=np.uint)


def bcommute(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Array of 0 for commutes and 1 for anticommutes bvectors."""

    # Convert to arrays.
    a = np.array(a)
    b = np.array(b)

    # If only singles, then convert to 2D array.
    if len(a.shape) == 1:
        a = np.reshape(a, (1, a.shape[0]))
    if len(b.shape) == 1:
        b = np.reshape(b, (1, b.shape[0]))

    # Check the shapes are correct.
    if a.shape[1] % 2 != 0:
        raise ValueError(
            f'Length {a.shape[1]} binary vector not of even length.'
        )
    if b.shape[1] % 2 != 0:
        raise ValueError(
            f'Length {b.shape[1]} binary vector not of even length.'
        )
    if b.shape[1] != a.shape[1]:
        raise ValueError(
            f'Length {a.shape[1]} bvector cannot be '
            'composed with length {b.shape[1]}'
        )

    # Number of qubits.
    n = int(a.shape[1]/2)

    # Commute commutator by binary symplectic form.
    commutes = np.zeros((a.shape[0], b.shape[0]), dtype=np.uint)
    for i_a in range(a.shape[0]):
        for i_b in range(b.shape[0]):
            a_X = a[i_a, :n]
            a_Z = a[i_a, n:]
            b_X = b[i_b, :n]
            b_Z = b[i_b, n:]
            commutes[i_a, i_b] = np.sum(a_X*b_Z + a_Z*b_X) % 2
    return commutes


def pauli_string_to_bvector(pauli_string: str) -> np.ndarray:
    X_block = []
    Z_block = []
    for character in pauli_string:
        if character == 'I':
            X_block.append(0)
            Z_block.append(0)
        elif character == 'X':
            X_block.append(1)
            Z_block.append(0)
        elif character == 'Y':
            X_block.append(1)
            Z_block.append(1)
        elif character == 'Z':
            X_block.append(0)
            Z_block.append(1)
    bvector = np.concatenate([X_block, Z_block]).astype(np.uint)
    return bvector


def bvector_to_pauli_string(bvector: np.ndarray) -> str:
    n = int(bvector.shape[0]/2)
    pauli_string = ''
    for i in range(n):
        pauli_string += {
            (0, 0): 'I',
            (1, 0): 'X',
            (1, 1): 'Y',
            (0, 1): 'Z'
        }[(bvector[i], bvector[i + n])]
    return pauli_string


def get_effective_error(
    logicals: np.ndarray,
    total_error: np.ndarray
) -> np.ndarray:
    """Effective Pauli error on logical qubits after decoding."""

    if len(logicals.shape) == 1:
        raise ValueError('Must have at least two logicals.')

    n_logical = int(len(logicals)/2)
    n_physical = int(logicals.shape[1]/2)
    assert n_physical > n_logical

    num_total_errors = 1
    if len(total_error.shape) > 1:
        num_total_errors = total_error.shape[0]

    # First half are X logicals, second half are Z logicals.
    X_logicals = np.array(logicals)[:n_logical]
    Z_logicals = np.array(logicals)[n_logical:]

    effective_Z = bcommute(X_logicals, total_error)
    effective_X = bcommute(Z_logicals, total_error)
    effective = np.concatenate([effective_X.T, effective_Z.T], axis=1)
    if num_total_errors == 1:
        effective = effective.reshape(2*n_logical)
    return effective


def bvector_to_int(bvector: np.ndarray) -> int:
    """Convert bvector to integer for effecient storage."""
    return int(''.join(map(str, bvector)), 2)


def int_to_bvector(int_rep: int, n: int) -> np.ndarray:
    """Convert integer representation to n-qubit Pauli bvector."""
    binary_string = ('{:0%db}' % (2*n)).format(int_rep)
    bvector = np.array(tuple(binary_string), dtype=np.uint)
    return bvector


def bvectors_to_ints(bvector_list: list) -> list:
    """List of bvectors to integers for efficient storage."""
    return list(map(
        bvector_to_int,
        bvector_list
    ))


def ints_to_bvectors(int_list: list, n: int) -> list:
    """Convert list of integers back to bvectors."""
    bvectors = []
    for int_rep in int_list:
        bvectors.append(int_to_bvector(int_rep, n))
    return bvectors
