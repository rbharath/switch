from numpy import *

def log_multivariate_normal_pdf(x, mu, Sigma):
    size = len(x)
    if size == len(mu) and (size, size) == shape(Sigma):
      det = linalg.det(Sigma)
      if det == 0:
        raise NameError("The covariance matrix can't be singular")

      try:
        log_norm_const = -0.5 * (float(size) * log(2*pi) + log(det))
      except FloatingPointError:
        log_norm_const = -Inf

      x_mu = x - mu
      inv = linalg.pinv(Sigma)
      log_result = -0.5 * dot(x_mu, dot(inv, x_mu.T))
      return log_norm_const + log_result
    else:
      raise NameError("The dimensions of the input don't match")

def logsumexp(x, dim=-1):
    """Compute log(sum(exp(x))) in a numerically stable way.

       Use second argument to specify along which dimensions the logsumexp
       shall be computed. If -1 (which is the default), logsumexp is
       computed along the last dimension.
    """
    if len(x.shape) < 2:
        xmax = x.max()
        return xmax + log(sum(exp(x-xmax)))
    else:
        if dim != -1:
            x = x.transpose(range(dim) + range(dim+1, len(x.shape)) + [dim])
        lastdim = len(x.shape)-1
        xmax = x.max(lastdim)
        return xmax + log(sum(exp(x-xmax[...,newaxis]),lastdim))
