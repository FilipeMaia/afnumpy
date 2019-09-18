import numpy as np


def isintlike(x):
    """Is x appropriate as an index into a sparse matrix? Returns True
    if it can be cast safely to a machine int.
    """
    # Fast-path check to eliminate non-scalar values. operator.index would
    # catch this case too, but the exception catching is slow.
    if np.ndim(x) != 0:
        return False
    try:
        operator.index(x)
    except (TypeError, ValueError):
        try:
            loose_int = bool(int(x) == x)
        except (TypeError, ValueError):
            return False
        if loose_int:
            warnings.warn("Inexact indices into sparse matrices are deprecated",
                          DeprecationWarning)
        return loose_int
    return True

def isshape(x, nonneg=False):
    """Is x a valid 2-tuple of dimensions?

    If nonneg, also checks that the dimensions are non-negative.
    """
    try:
        # Assume it's a tuple of matrix dimensions (M, N)
        (M, N) = x
    except:
        return False
    else:
        if isintlike(M) and isintlike(N):
            if np.ndim(M) == 0 and np.ndim(N) == 0:
                if not nonneg or (M >= 0 and N >= 0):
                    return True
        return False