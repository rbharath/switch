from cvxopt import matrix, solvers
# Define constants
a = 0.9
d = 2
v = 1

c = matrix([1., 0.])
G = [matrix([[-1., 0.],
             [0.,  0.],
             [0.,  0.],
             [0.,  0.],
             [0.,  0.], # First column
             [0.,  0.],
             [0., -1.],
             [0.,  0.],
             [0.,  0.],
             [0.,  0.], # Second Column
             [0.,  0.],
             [0.,  0.],
             [0.,  1.],
             [0.,  0.],
             [0.,  0.], # Third Column
             [0.,  0.],
             [0.,  0.],
             [0.,  0.],
             [0.,  0.],
             [0.,  0.], # Fourth Column
             [0.,  0.],
             [0.,  0.],
             [0.,  0.],
             [0.,  0.],
             [0.,  0.]])] # Fifth Column
G = [G[0].T]

h = [matrix([[0.,  v, 0.,     0., 0.],
             [ v, 0., 0.,     0., 0.],
             [ 0, 0.,  d,      a, 0.],
             [ 0, 0.,  a, (1./d), 0.],
             [ 0, 0., 0.,     0., 0.]])]
h = [h[0].T]

sol = solvers.sdp(c, Gs = G, hs=h)
