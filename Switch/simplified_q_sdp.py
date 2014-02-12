from cvxopt import matrix, solvers
from numpy import bmat
from numpy.linalg import pinv
# Define constants
a = 0.9
d = 2
y = 1.
x = 1.
b = 0.5

def solve_Q(x, b, y, A, D):
  c = matrix([1., 0.])
  # x = [t vec(Q)]
  xdim = len(x)
  gdim = 1 + 4 * xdim
  G = [zeros((gdim, gdim))]
  G = [array([[-1., 0.],
              [0.,  0.],
              [0.,  0.],
              [0.,  0.],
              [0.,  0.],
              [0.,  0.],
              [0.,  0.],
              [0.,  0.],
              [0.,  0.], # First column
              [0.,  0.],
              [0., -1.],
              [0.,  0, -1.,   0,   0, 0.],
              [0.,  0,   0, -1.,   0, 0.],
              [0.,  0,   0,  0.,  0., 0.],
              [0.,  0,   0,  0.,  0., 0.],
              [0.,  0.],
              [0.,  0.],
              [0.,  0.], # Second Column
              [0.,  0.],
              [0.,  0.],
              [0.,  1.],
              [0.,  0.],
              [0.,  0.], # Third Block Column
              [0.,  0.],
              [0.,  0.],
              [0.,  0.],
              [0.,  0.],
              [0.,  0.], # Fourth Block Column
              [0.,  0.],
              [0.,  0.],
              [0.,  0.],
              [0.,  0.],
              [0., -1.]])] # Fifth Block Column
  G = [matrix(G[0])]

  h = [array([[        0.,  y-(a*x+b),  0.,      0., 0.],
              [ y-(a*x+b),          0., 0.,      0., 0.],
              [         0,          0.,  D,       A, 0.],
              [         0,          0.,  A, pinv(D), 0.],
              [         0,          0., 0.,      0., 0.]])]
  h = [matrix(h[0])]

  sol = solvers.sdp(c, Gs = G, hs=h)
  return sol

solve_Q(x, b, y, a, d)
