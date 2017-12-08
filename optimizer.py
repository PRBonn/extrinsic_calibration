#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:53:59 2017

@author: kaihong
"""
import numpy as np

def huber_w(y):
    return np.fmin(1, 1/np.abs(y))

def huber_w_smooth(y):
    return 1/np.sqrt(1 + y**2)

def exp_w(c=2):
    def _exp_w(y):
        return np.exp(-0.5*y**2/c**2)
    return _exp_w

def SolveGaussHelmertProblem(g, x, lm, Cov_ll, maxiter=100, thres=1e-5):
    """ Solve Gauss-Helmert Problem:
      argmin ||dl||^2, subject to g(x+dx, l+dl)=0, where x is unknown parameters
      and l are observations.

    Input
    --------
    g: callable object
      Represent constraint function g(x,l)=0.
      f should take two arguments (x and l) and return a tuple (error, Jx, Jl),
      where error is g(x,l), Jx is the Jacobian dg/dx and Jl is the Jacobian dg/dl.

    x: 1xP array
      Parameter initial guess x0 of size P

    lm: NxQ array
      Observation matrix, each row represent one observation l instance of size Q

    Cov_ll: NxQxQ array
      Covariance matrice for each observation instance l1 ... lN

    maxiter: int
      Terminate when maximum number of iteration reached.

    thres: float
      Terminate when max(|dx|) < thres

    Return
    --------
    xnu: 1xP array
      Estimated parameter vector x

    Cov_xx: PxP array
      Covariance matrix of estimated x

    sigma_0: float
      factor, should close 1

    vv: NxQ array
      Observation corrections dl

    w: 1xN array
      Weights for each observation l, lower weight means an outliar
    """

    N, Q = lm.shape[:2]
    err_i, A_i, B_i = g(x, lm[0,:])
    M = len(err_i)  # residual size
    P = len(x)      # parameter size

    R  = N * Q - P  # estimation redundancy
    if R<0:
        raise RuntimeError('Not enough observations')

    ''' init '''
    xnu = x.copy()
    lnu = lm.copy()                 # current updated observations
    vv = np.zeros_like(lm)          # observations correction

    Am   = np.empty( (N,) + A_i.shape )
    Bm   = np.empty( (N,) + B_i.shape )
    W_gg = np.empty( (N,) + (M, M) )
    Cg   = np.empty( (N, M)  )
    Nm   = np.empty( (P, P)  )
    nv   = np.empty( P )
    X_gg = np.empty( N )

    W_ll = np.empty_like(Cov_ll)
    for i in xrange(N):
      W_ll[i] = np.linalg.pinv(Cov_ll[i])

    for it in xrange(maxiter):
        Nm[:,:] = 0
        nv[:] = 0
        for i in xrange(N):
            # update Jacobian and residual
            err_i, Am[i], Bm[i] = g(xnu, lnu[i,:])
            A_i, B_i = Am[i], Bm[i]
            Cg[i] = B_i.dot(vv[i]) - err_i

            # weights of constraints
            W_gg[i] = np.linalg.pinv( B_i.dot(Cov_ll[i]).dot(B_i.T) )

            X_gg[i] = np.sqrt( Cg[i].T.dot(W_gg[i]).dot(Cg[i]) )
        sigma_gg = np.median(X_gg)/0.6745
        w = huber_w_smooth( X_gg/sigma_gg )

        for i in xrange(N):
            # normal equation
            ATBWB = Am[i].T.dot(w[i]*W_gg[i])
            Nm  += ATBWB.dot(Am[i])
            nv  += ATBWB.dot(Cg[i])


        lambda_N = np.sort( np.linalg.eigvalsh(Nm) )
        if np.abs(np.log( lambda_N[-1]/lambda_N[0] ) ) > 10:
            print( 'normal equation matrix nearly singular')

        # solve
        dx = np.linalg.solve(Nm, nv)

        # update
        xnu = xnu + dx

        omega = 0
        for i in xrange(N):
            lamba = W_gg[i].dot( Am[i].dot(dx) - Cg[i] )
            dl    = -Cov_ll[i].dot( Bm[i].T.dot(lamba) ) - vv[i]

            lnu[i,:] = lnu[i,:] + dl

            # calculate observation-corrections
            vv[i] = lnu[i,:] - lm[i,:]

        Cov_xx  = np.linalg.pinv(Nm)
        omega = np.sum([vv[i].dot(W_ll[i]).dot(vv[i]) for i in xrange(N)])
        sigma_0  = omega/R
        print('GH Iter %d: %f' % (it, sigma_0))

        if np.abs(dx).max() < thres:
            break
    return xnu, Cov_xx, sigma_0, vv, w

def test_SolveGaussHelmertProblem():
    '''' implicit test model g(x,l) = x*l = 10 '''
    def g(x,l):
        return np.atleast_1d(np.dot(x,l)-10), np.atleast_2d(l), np.atleast_2d(x)
    x_true = np.array([1.,1.,1.])

    # data
    sigma_l = 1e-3
    sigma_x = 1e-2
    T = 500

    lm = np.empty((T,3))
    x0 = x_true + sigma_x*np.random.randn(3)
    l_true = np.empty_like(x0)
    for t in xrange(T):
      l_true[:2] = 3*np.random.randn(2)
      l_true[2]  = 10 - np.sum(l_true[:2])
      lm[t,:] = l_true + sigma_l*np.random.randn(3)
    Cov_ll = np.tile( sigma_l**2*np.eye(3), (T,1,1) )
    # solve
    xe, Cov_xx, sigma_0, vv, w = SolveGaussHelmertProblem(g, x0, lm, Cov_ll)
    np.testing.assert_array_almost_equal(x_true, xe, 4)
    print('test: solving implicit test model passed')


    '''explict test model f(x,l): A*x=l'''
    A = np.diag([3.,2.,1.])
    x_true = np.array([1.,2.,3.])
    l_true = A.dot(x_true)
    def f(x, l):
        return (A.dot(x)-l, A, -np.eye(len(l)))


    for t in xrange(T):
      lm[t,:] = l_true + sigma_l*np.random.randn(3)
    Cov_ll = np.tile( sigma_l**2*np.eye(3), (T,1,1) )

    x0 = x_true + sigma_x*np.random.randn(3)

    # solve
    xe, Cov_xx, sigma_0, vv, w = SolveGaussHelmertProblem(f, x0, lm, Cov_ll)
    np.testing.assert_array_almost_equal(x_true, xe, 4)
    print('test: solving explicit test model passed')

    # batch test
    ntest = 100
    sigmas = np.empty(ntest)
    for i in range(ntest):
      x0 = x_true + sigma_x*np.random.randn(3)
      for t in xrange(T):
            lm[t,:] = l_true + sigma_l*np.random.randn(3)
      xe, Cov_xx, sigmas[i], vv, w = SolveGaussHelmertProblem(f, x0, lm, Cov_ll)
    print("Mean sigma_0: %f (which should be close to 1)" % np.mean(sigmas))


if __name__ == "__main__":
  test_SolveGaussHelmertProblem()

