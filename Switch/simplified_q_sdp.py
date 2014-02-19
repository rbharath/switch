from cvxopt import matrix, solvers
from numpy import bmat, zeros, reshape, array, dot, shape, eye, shape
from numpy.linalg import pinv
from scipy.linalg import block_diag
# Define constants
a = array([[0.9]])
d = array([[2]])
#y = array([1.])
xs = array([1., 1.])
b = array([0.5])
v = xs[1] - dot(a,xs[0]) - b
v = reshape(v, (len(v),1))
B = dot(v,v.T)

def construct_coeff_matrix(x_dim):
  # x = [diag(T) vec(Q)]
  # T is diagonal = [t1,...,tn]
  g_dim = 5 * x_dim
  G = zeros((g_dim**2, x_dim + x_dim **2))
  # --------------------
  #|T  I
  #|B  Q
  #|      D-Q    A
  #|      A.T  D^{-1}
  #|                   Q
  # --------------------

  # First Block Column
  # T
  for i in range(x_dim): # row/col on diag
    vec_pos = i #pos in param vector
    mat_pos = i * g_dim + i
    G[mat_pos, vec_pos] = 1.
  # Second Block Column
  left = x_dim
  top = x_dim
  for j in range(x_dim): # cols
    for i in range(x_dim): # rows
      vec_pos = x_dim + j * x_dim + i #pos in param vector
      mat_pos = left * g_dim + j * g_dim + top + i
      G[mat_pos, vec_pos] = 1.
  # Third Block Column
  left = 2 * x_dim
  top = 2 * x_dim
  for j in range(x_dim): # cols
    for i in range(x_dim): # rows
      vec_pos = 1 + j * x_dim + i #pos in param vector
      mat_pos = left * g_dim + j * g_dim + top + i
      G[mat_pos, vec_pos] = -1.
  # Fifth Block Column
  left = 4 * x_dim
  top = 4 * x_dim
  for j in range(x_dim): # cols
    for i in range(x_dim): # rows
      vec_pos = x_dim + j * x_dim + i #pos in param vector
      mat_pos = left * g_dim + j * g_dim + top + i
      G[mat_pos, vec_pos] = 1.
  return G

def construct_const_matrix(A, B, D):
  #[[ 0, I, 0,       0, 0],
  # [ B, 0, 0,       0, 0],
  # [ 0, 0, D,       A, 0],
  # [ 0, 0, A, pinv(D), 0],
  # [ 0, 0, 0,       0, 0]]
  x_dim = shape(A)[0]
  # Construct B1
  B1 = zeros((2 * x_dim, 2 * x_dim))
  #x = reshape(x, (x_dim, 1))
  #v = y - dot(A,x)
  #B1[0,1:] = v.T
  #B1[1:,0] = v
  B1[x_dim:,0:x_dim] = B
  B1[0:x_dim,x_dim:] = eye(x_dim)

  # Construct B2
  B2 = zeros((2 * x_dim, 2 * x_dim))
  B2[0:x_dim, 0:x_dim] = D
  B2[0:x_dim, x_dim:2*x_dim] = A
  B2[x_dim:2*x_dim,0:x_dim] = A.T
  B2[x_dim:2*x_dim,x_dim:2*x_dim] = pinv(D)

  # Construct B3
  B3 = zeros((x_dim, x_dim))

  # Construct Block matrix
  h = block_diag(B1, B2, B3)
  return h

def solve_Q(A, B, D):
  x_dim = shape(A)[0]
  c = matrix([1., 0.])
  G = -construct_coeff_matrix(x_dim)
  print "G-shape"
  print shape(G)
  Gs = [matrix(G)]

  h = construct_const_matrix(A, B, D)
  print "h-shape"
  print shape(h)
  hs = [matrix(h)]

  sol = solvers.sdp(c, Gs = Gs, hs=hs)
  print sol['x']
  return sol

solve_Q(a, b, d)
