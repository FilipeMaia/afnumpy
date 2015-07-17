import afnumpy
from numpy import result_type

def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None):
    num = int(num)

    # Convert float/complex array scalars to float, gh-3504 
    start = start * 1.
    stop = stop * 1.

    if dtype is None:
        dtype = result_type(start, stop, float(num))

    if num <= 0:
        return array([], dtype)
    if endpoint:
        if num == 1:
            return array([start], dtype=dtype)
        step = (stop-start)/float((num-1))
        y = afnumpy.arange(0, num, dtype=dtype) * step + start
        y[-1] = stop
    else:
        step = (stop-start)/float(num)
        y = afnumpy.arange(0, num, dtype=dtype) * step + start
    if retstep:
        return y.astype(dtype), step
    else:
        return y.astype(dtype)
