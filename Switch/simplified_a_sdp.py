from cvxopt import matrix, solvers
from numpy import bmat, zeros, reshape, array, dot, eye
from numpy.linalg import pinv
from scipy.linalg import block_diag

# Define constants
x = array([1.1])
b = array([0.5])
y = array([1.])
q = array([[1.]])
d = array([[2.]])
x_dim = len(x)

def construct_coeff_matrix(x_dim, x):
  # x = [t vec(A)]
  g_dim = 1 + 5 * x_dim
  G = zeros((g_dim**2, 1 + x_dim **2))
  # -------------------------
  #|t v.T
  #|v  Q
  #|      D-Q    A
  #|      A.T  D^{-1}
  #|                  I-A
  #|                      A+I
  # -------------------------

  # First Block Column
  # t
  G[0,0] = 1.
  # -Ax
  left = 0
  top = 1
  for i in range(x_dim): # rows
    mat_pos = left * g_dim + top + i
    for j in range(x_dim): # cols
      vec_pos = 1 + j * x_dim + i #pos in param vector
      G[mat_pos, vec_pos] = -x[j]
  # Second Block Column
  # (-Ax).T
  left = 1
  top = 0
  for i in range(x_dim): # cols G / rows A
    mat_pos = left * g_dim + i * g_dim + top
    for j in range(x_dim): # cols A
      vec_pos = 1 + j * x_dim + i # pos in param vector
      G[mat_pos, vec_pos] = -x[j]
  # Third Block Column
  # A.T
  left = 1 + x_dim
  top = 1 + 2 * x_dim
  for j in range(x_dim): # cols
    for i in range(x_dim): # rows
      vec_pos = 1 + i * x_dim + j #pos in param vector
      mat_pos = left * g_dim + j * g_dim + top + i
      G[mat_pos, vec_pos] = 1.
  # Fourth Block Column
  # A
  left = 1 + 2 * x_dim
  top = 1 + x_dim
  for j in range(x_dim): # cols
    for i in range(x_dim): # rows
      vec_pos = 1 + j * x_dim + i #pos in param vector
      mat_pos = left * g_dim + j * g_dim + top + i
      G[mat_pos, vec_pos] = 1.
  # Fifth Block Column
  left = 1 + 3 * x_dim
  top = 1 + 3 * x_dim
  for j in range(x_dim): # cols
    for i in range(x_dim): # rows
      vec_pos = 1 + j * x_dim + i #pos in param vector
      mat_pos = left * g_dim + j * g_dim + top + i
      G[mat_pos, vec_pos] = -1.
  # Fifth Block Column
  left = 1 + 4 * x_dim
  top = 1 + 4 * x_dim
  for j in range(x_dim): # cols
    for i in range(x_dim): # rows
      vec_pos = 1 + j * x_dim + i #pos in param vector
      mat_pos = left * g_dim + j * g_dim + top + i
      G[mat_pos, vec_pos] = 1.
  return G

def construct_const_matrix(x_dim, y, x, D, Q):
  #[    y                      ]
  #[ y  Q                      ]
  #[         D-Q        A      ]
  #[         A.T  pinv(D)      ]
  #[                       I   ]
  #[                          I]
  # Construct B1
  B1 = zeros((1+x_dim, 1+x_dim))
  x = reshape(x, (x_dim, 1))
  v = y
  B1[0,1:] = v.T
  B1[1:,0] = v
  B1[1:,1:] = Q

  # Construct B2
  B2 = zeros((2 * x_dim, 2 * x_dim))
  B2[0:x_dim, 0:x_dim] = D - Q
  B2[x_dim:2*x_dim,x_dim:2*x_dim] = pinv(D)

  # Construct B3
  B3 = eye(x_dim)

  # Construct B4
  B4 = eye(x_dim)

  # Construct Block matrix
  h = block_diag(B1, B2, B3, B4)
  return h

def solve_A(x, b, y, Q, D):
  c = matrix([1., 0.])
  #G = [matrix([[-1.,  0.],
  #             [ 0.,   x],
  #             [ 0.,  0.],
  #             [ 0.,  0.],
  #             [ 0.,  0.],
  #             [ 0.,  0.], # First column
  #             [ 0.,   x],
  #             [ 0.,  0.],
  #             [ 0.,  0.],
  #             [ 0.,  0.],
  #             [ 0.,  0.],
  #             [ 0.,  0.], # Second Column
  #             [ 0.,  0.],
  #             [ 0.,  0.],
  #             [ 0.,  0.],
  #             [ 0., -1.],
  #             [ 0.,  0.],
  #             [ 0.,  0.], # Third Column
  #             [ 0.,  0.],
  #             [ 0.,  0.],
  #             [ 0., -1.],
  #             [ 0.,  0.],
  #             [ 0.,  0.],
  #             [ 0.,  0.], # Fourth Column
  #             [ 0.,  0.],
  #             [ 0.,  0.],
  #             [ 0.,  0.],
  #             [ 0.,  0.],
  #             [ 0.,  1.],
  #             [ 0.,  0.], # Fifth Column
  #             [ 0.,  0.],
  #             [ 0.,  0.],
  #             [ 0.,  0.],
  #             [ 0.,  0.],
  #             [ 0.,  0.],
  #             [ 0., -1.]])] # Sixth Column
  G = -construct_coeff_matrix(x_dim, x)
  print "G"
  print G
  G = [matrix(G)]

  #h = [matrix([[ 0., y-b,  0.,     0., 0., 0.],
  #             [y-b,   Q,  0.,     0., 0., 0.],
  #             [ 0,   0., D-Q,     0., 0., 0.],
  #             [ 0,   0.,  0., (1./D), 0., 0.],
  #             [ 0,   0.,  0.,     0., 1., 0.],
  #             [ 0,   0.,  0.,     0., 0., 1.]])]
  h = construct_const_matrix(x_dim, y, x, D, Q)
  print "h"
  print h
  h = [matrix(h)]

  sol = solvers.sdp(c, Gs = G, hs=h)
  print sol['x']
  return sol
solve_A(x, b, y, q, d)
