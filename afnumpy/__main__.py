from multiarray import ndarray, zeros, ones, where, array
from . import random

a = random.rand(3)
#a = ones(3)
print a

print a+a
print 3+a
print a+3

print a-a
print a-3
print 3-a

print a*a
print a*3
print 3*a

print a/a
print a/3
print 3/a

print a**3
print a**a

print a < a
print a < 3
print 3 < a

print a <= a
print a <= 3
print 3 <= a

print a > a
print a > 3
print 3 > a

print a >= a
print a >= 3
print 3 >= a

print a == a
print a == 3
print 3 == a

print a != a
print a != 3
print 3 != a

a += a
print a
a += 3
print a

a -= a
print a
a -= 3
print a

a *= a
print a
a *= 3
print a

a /= a
print a
a /= 3
print a

print a.__nonzero__()
print where(a)
b = array(a)
print b

c = random.rand(3)
print c
d = array([1.,2.,0.])
print c.dtype
print d.dtype

print a[a]
print c[a]
e = (c > 0.5)
print e.dtype
print c[(c > 0.5)]
print c[d]
#print d[a]

#f = random.rand(3,3)
#print 2*f

