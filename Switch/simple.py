from cvxopt import matrix, solvers
from numpy import bmat, zeros, reshape, array, dot, eye, outer, shape
from numpy import amin
from numpy.linalg import pinv, eig
from numpy.random import rand
from scipy.linalg import block_diag

dim = 100
B = rand(dim,dim)
B = dot(B.T,B)
C = rand(dim,dim)
C = dot(C.T,C)
BC = dot(B,C)
m = amin(eig(BC)[0])
print m
print m >= 0


