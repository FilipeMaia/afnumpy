from .base import spmatrix,isspmatrix
from .sputils import isshape
import arrayfire
import afnumpy as afnp
from .. import private_utils as pu

class csr_matrix(spmatrix):
    """
    Compressed Sparse Row matrix

    This can be instantiated in several ways:
        csr_matrix(D)
            with a dense matrix or rank-2 ndarray D

        csr_matrix(S)
            with another sparse matrix S (equivalent to S.tocsr())

        csr_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
            where ``data``, ``row_ind`` and ``col_ind`` satisfy the
            relationship ``a[row_ind[k], col_ind[k]] = data[k]``.

        csr_matrix((data, indices, indptr), [shape=(M, N)])
            is the standard CSR representation where the column indices for
            row i are stored in ``indices[indptr[i]:indptr[i+1]]`` and their
            corresponding values are stored in ``data[indptr[i]:indptr[i+1]]``.
            If the shape parameter is not supplied, the matrix dimensions
            are inferred from the index arrays.

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of nonzero elements
    data
        CSR format data array of the matrix
    indices
        CSR format index array of the matrix
    indptr
        CSR format index pointer array of the matrix
    has_sorted_indices
        Whether indices are sorted

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the CSR format
      - efficient arithmetic operations CSR + CSR, CSR * CSR, etc.
      - efficient row slicing
      - fast matrix vector products

    Disadvantages of the CSR format
      - slow column slicing operations (consider CSC)
      - changes to the sparsity structure are expensive (consider LIL or DOK)

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> csr_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> row = np.array([0, 0, 1, 2, 2, 2])
    >>> col = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])

    As an example of how to construct a CSR matrix incrementally,
    the following snippet builds a term-document matrix from texts:

    >>> docs = [["hello", "world", "hello"], ["goodbye", "cruel", "world"]]
    >>> indptr = [0]
    >>> indices = []
    >>> data = []
    >>> vocabulary = {}
    >>> for d in docs:
    ...     for term in d:
    ...         index = vocabulary.setdefault(term, len(vocabulary))
    ...         indices.append(index)
    ...         data.append(1)
    ...     indptr.append(len(indices))
    ...
    >>> csr_matrix((data, indices, indptr), dtype=int).toarray()
    array([[2, 1, 0, 0],
           [0, 1, 1, 1]])

    """
    format = 'csr'

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        spmatrix.__init__(self)

        if isspmatrix(arg1):
            raise ValueError("Copying another sparse matrix not yet implemented")

        elif isinstance(arg1, tuple):
            if isshape(arg1):
                raise ValueError("Empty sparse matrices not implemented")
                # It's a tuple of matrix dimensions (M, N)
                # create empty matrix
                # self._shape = check_shape(arg1)
            else:
                if len(arg1) == 2:                    
                    try:
                        obj, (row, col) = arg1
                    except (TypeError, ValueError):
                        raise TypeError('invalid input format')
                    if shape is None:
                        raise ValueError("shape parameter must not be None") 
                    obj = afnp.asarray(obj)
                    # Arrayfire only supports floating point values
                    if obj.dtype.kind != 'c' and obj.dtype.kind != 'f':
                        obj = obj.astype(afnp.float32)
                    self.dtype = obj.dtype
                    self.d_array = arrayfire.sparse.create_sparse(afnp.asarray(obj).d_array,
                                                             afnp.asarray(row).astype(afnp.int32).d_array, 
                                                             afnp.asarray(col).astype(afnp.int32).d_array,
                                                             shape[0], shape[1],  storage = arrayfire.STORAGE.COO)
                    self.d_array = arrayfire.convert_sparse(self.d_array, arrayfire.STORAGE.CSR)
                    self._shape = shape
                elif len(arg1) == 3:
                    raise ValueError("(data, indices, indptr) format not implemented")
                else:
                    raise ValueError("unrecognized %s_matrix constructor usage" %
                            self.format)

        else:
            # must be dense
            try:
                arg1 = afnp.asarray(arg1)
            except:
                raise ValueError("unrecognized %s_matrix constructor usage" %
                        self.format)
            d_array = arrayfire.sparse.create_sparse_from_dense(arg1)

    def __mul__(self, other):
        other = afnp.asarray(other).astype(self.dtype)
        s = arrayfire.matmul(self.d_array, other.d_array)
        a = afnp.ndarray(pu.af_shape(s), dtype=pu.typemap(s.dtype()), af_array=s)
        a._eval()
        return a