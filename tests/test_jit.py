import afnumpy
from asserts import *

def test_comparisons():
    a = afnumpy.arange(10, dtype="float32") - 5.
    b = afnumpy.ones((10), dtype="float32")
    afnumpy.arrayfire.backend.get().af_eval(a.d_array.arr)
    a_mask = a < 0.
    a_sum = a_mask.sum()
    a -= b
    assert(a_sum == a_mask.sum())
    a_mask = a > 0.
    a_sum = a_mask.sum()
    a -= b
    assert(a_sum == a_mask.sum())
    a_mask = a >= 0.
    a_sum = a_mask.sum()
    a -= b
    assert(a_sum == a_mask.sum())
    a_mask = a <= 0.
    a_sum = a_mask.sum()
    a -= b
    assert(a_sum == a_mask.sum())
    a_mask = a == 0.
    a_sum = a_mask.sum()
    a *= 0
    assert(a_sum == a_mask.sum())
    a = afnumpy.arange(10, dtype="float32") - 5.
    afnumpy.arrayfire.backend.get().af_eval(a.d_array.arr)
    a_mask = a != 0.
    a_sum = a_mask.sum()
    a *= 0
    assert(a_sum == a_mask.sum())

def test_unary():
    a = afnumpy.arange(10, dtype="float32")-5
    afnumpy.arrayfire.backend.get().af_eval(a.d_array.arr)
    c = -a
    c_sum = c.sum()
    a *= -1
#    assert(c_sum == c.sum())
    c = +a
    c_sum = c.sum()
    a *= -1
    assert(c_sum == c.sum())
