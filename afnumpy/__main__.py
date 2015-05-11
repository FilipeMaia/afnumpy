from multiarray import ndarray, zeros, ones
from . import random

a = ones(3)
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



#f = random.rand(3,3)
#print 2*f

