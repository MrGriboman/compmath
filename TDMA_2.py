from scipy.linalg import solve_banded

def TDMA(matrix, rhs):

    banded_matrix = np.zeros((3, len(matrix)))

    for offset in range(1, -2, -1):

        start = offset if offset > 0 else None

        end = offset if offset < 0 else None

        banded_matrix[1 - offset, start:end] = np.diag(matrix, offset)

    return solve_banded((1, 1), banded_matrix, rhs)