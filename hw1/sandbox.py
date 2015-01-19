import sys
import numpy

m = numpy.array([[1, 2], [3, 4]])
n = numpy.array([[7, 13], [17, 19]])
print 'm:'
print m
print 'n:'
print n
print 'm * n:'
print m * n
print 'm.dot(n)'
print m.dot(n)
print m * n[:,:,numpy.newaxis]
