import afnumpy
import numpy

def broadcast_arrays(*args, **kwargs):
    subok = kwargs.pop('subok', False)
    if kwargs:
        raise TypeError('broadcast_arrays() got an unexpected keyword '
                        'argument {}'.format(kwargs.pop()))
    args = [afnumpy.array(_m, copy=False, subok=subok) for _m in args]
    shapes = [x.shape for x in args]
    if len(set(shapes)) == 1:
        # Common case where nothing needs to be broadcasted.
        return args
    shapes = [list(s) for s in shapes]
    strides = [list(x.strides) for x in args]
    nds = [len(s) for s in shapes]
    biggest = max(nds)
    # Go through each array and prepend dimensions of length 1 to each of
    # the shapes in order to make the number of dimensions equal.
    for i in range(len(args)):
        diff = biggest - nds[i]
        if diff > 0:
            shapes[i] = [1] * diff + shapes[i]
            strides[i] = [0] * diff + strides[i]
    # Chech each dimension for compatibility. A dimension length of 1 is
    # accepted as compatible with any other length.
    common_shape = []
    for axis in range(biggest):
        lengths = [s[axis] for s in shapes]
        unique = set(lengths + [1])
        if len(unique) > 2:
            # There must be at least two non-1 lengths for this axis.
            raise ValueError("shape mismatch: two or more arrays have "
                "incompatible dimensions on axis %r." % (axis,))
        elif len(unique) == 2:
            # There is exactly one non-1 length. The common shape will take
            # this value.
            unique.remove(1)
            new_length = unique.pop()
            common_shape.append(new_length)
            # For each array, if this axis is being broadcasted from a
            # length of 1, then set its stride to 0 so that it repeats its
            # data.
            for i in range(len(args)):
                if shapes[i][axis] == 1:
                    shapes[i][axis] = new_length
                    strides[i][axis] = 0
        else:
            # Every array has a length of 1 on this axis. Strides can be
            # left alone as nothing is broadcasted.
            common_shape.append(1)

    # Construct the new arrays.
    broadcasted = []

    for (x, sh) in zip(args, shapes):
        x_sh = x.shape + (1,)*(len(sh)-x.ndim)
        reps = numpy.array(sh)/numpy.array(x_sh)
        if(numpy.prod(reps) > 1):
            broadcasted.append(afnumpy.tile(x, reps))
        else:
            if(x.shape != tuple(sh)):
                x = x.reshape(sh)
            broadcasted.append(x)
            

    return broadcasted

