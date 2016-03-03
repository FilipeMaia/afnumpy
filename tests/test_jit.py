import afnumpy
from asserts import *
import pytest
from decorators import *
xfail = pytest.mark.xfail

@foreach_backend
def test_comparisons():
    a = afnumpy.arange(10, dtype="float32") - 5.
    b = afnumpy.ones((10), dtype="float32")
    a.eval()
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
    a.eval()
    a_mask = a != 0.
    a_sum = a_mask.sum()
    a *= 0
    assert(a_sum == a_mask.sum())

@foreach_backend
def test_unary():
    a = afnumpy.arange(10, dtype="float32")-5
    a.eval()
    c = -a
    c_sum = c.sum()
    a *= -1
    assert(c_sum == c.sum())
    c = +a
    c_sum = c.sum()
    a *= -1
    assert(c_sum == c.sum())

@foreach_backend
def test_arithmetic():
    a = afnumpy.arange(10, dtype="float32")
    a.eval()
    c = a % 3
    c_sum = c.sum()
    a += 1
    assert(c_sum == c.sum())
    c = 3 % a
    c_sum = c.sum()
    a += 1
    assert(c_sum == c.sum())
    c = a**1
    c_sum = c.sum()
    a -= 1
    assert(c_sum == c.sum())
    c = 1.02**a
    c_sum = c.sum()
    a += 1
    assert(c_sum == c.sum())
    c = a+1
    c_sum = c.sum()
    a -= 1
    assert(c_sum == c.sum())
    c = 1+a
    c_sum = c.sum()
    a += 1
    assert(c_sum == c.sum())
    c = a-1
    c_sum = c.sum()
    a -= 1
    assert(c_sum == c.sum())
    c = 1-a
    c_sum = c.sum()
    a += 1
    assert(c_sum == c.sum())
    c = a*2
    c_sum = c.sum()
    a -= 1
    assert(c_sum == c.sum())
    c = 2*a
    c_sum = c.sum()
    a += 1
    assert(c_sum == c.sum())
    c = a/2
    c_sum = c.sum()
    a -= 1
    assert(c_sum == c.sum())
    c = 2/a
    c_sum = c.sum()
    a += 1
    assert(c_sum == c.sum())
