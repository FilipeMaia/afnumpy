import afnumpy

def copy(a, order='K'):
    return afnumpy.array(a, order=order, copy=True)
