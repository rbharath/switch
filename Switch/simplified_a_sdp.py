from cvxopt import matrix, solvers
# Define constants
x = 1.
b = 0.5
y = 1.
q = 1.
d = 2.

c = matrix([1., 0.])
G = [matrix([[-1.,  0.],
             [ 0.,   x],
             [ 0.,  0.],
             [ 0.,  0.],
             [ 0.,  0.],
             [ 0.,  0.], # First column
             [ 0.,   x],
             [ 0.,  0.],
             [ 0.,  0.],
             [ 0.,  0.],
             [ 0.,  0.],
             [ 0.,  0.], # Second Column
             [ 0.,  0.],
             [ 0.,  0.],
             [ 0.,  0.],
             [ 0., -1.],
             [ 0.,  0.],
             [ 0.,  0.], # Third Column
             [ 0.,  0.],
             [ 0.,  0.],
             [ 0., -1.],
             [ 0.,  0.],
             [ 0.,  0.],
             [ 0.,  0.], # Fourth Column
             [ 0.,  0.],
             [ 0.,  0.],
             [ 0.,  0.],
             [ 0.,  0.],
             [ 0.,  1.],
             [ 0.,  0.], # Fifth Column
             [ 0.,  0.],
             [ 0.,  0.],
             [ 0.,  0.],
             [ 0.,  0.],
             [ 0.,  0.],
             [ 0., -1.]])] # Sixth Column
G = [G[0].T]

h = [matrix([[ 0., y-b,  0.,     0., 0., 0.],
             [y-b,   q,  0.,     0., 0., 0.],
             [ 0,   0., d-q,     0., 0., 0.],
             [ 0,   0.,  0., (1./d), 0., 0.],
             [ 0,   0.,  0.,     0., 1., 0.],
             [ 0,   0.,  0.,     0., 0., 1.]])]
h = [h[0].T]

sol = solvers.sdp(c, Gs = G, hs=h)
