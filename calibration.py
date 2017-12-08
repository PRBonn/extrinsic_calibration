#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:58:46 2017

@author: kaihong
"""
from __future__ import print_function
import numpy as np
import scipy
from optimizer import SolveGaussHelmertProblem
from numpy.random import randn

from pycppad import independent,adfun
def GenerateAutoDiffFunction(g, x0, l0):
    dim_x = x0.shape[0]
    a_var = independent(np.hstack([x0, l0]))
    jacobian = adfun(a_var, g(a_var[:dim_x], a_var[dim_x:])).jacobian

    def g_autodiff(x, l):
        err = g(x, l)
        J   = jacobian(np.hstack([x, l]))
        return err, J[:, :dim_x], J[:, dim_x:]
    return g_autodiff


def Rot2ax(R):
    """Rotation matrix to angle-axis vector"""
    tr = np.trace(R)
    a  = np.array( [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]] )
    an = np.linalg.norm(a)
    phi= np.arctan2(an, tr-1)
    if np.abs(phi) < 1e-12:
        return np.zeros(3,'d')
    else:
        return phi/an*a

def skew(v):
    return np.array([[   0, -v[2],  v[1]],
                     [ v[2],    0, -v[0]],
                     [-v[1], v[0],    0 ]])

def ax2Rot(r):
    """Angle-axis vector to rotation matrix"""
    p = np.linalg.norm(r)
    if np.abs(p) < 1e-12:
        return np.eye(3)
    else:
        S = skew(r/p)
        return np.eye(3) + np.sin(p)*S + (1.0-np.cos(p))*S.dot(S)

def MfromRT(r,t):
    T = np.eye(4)
    T[:3,:3] = ax2Rot(r)
    T[:3, 3] = t
    return T

def RTfromM(mat):
    return Rot2ax(mat[:3,:3]), mat[:3,3]

#%%
def ExtrinsicCalibration3D(trajectories_list, trajectories_cov_list=None, *args):
    """ Motion-base sensor calibration
    Constrain equations:
        ( I3 - R(r_b) )*xi_b = t_b - R(eta_b)*t_a
    and
        R(eta_b)*r_a = r_b

    Input
    -----------
    trajectories_list: list( list( 4x4 pose matrices ) )

    Output
    -----------
    calibration result as a list of matrix
    """
    num_pose_list   = map(len, trajectories_list)
    if len(set(num_pose_list))!=1:
        raise ValueError("each trajectory should have the same number of poses")

    num_pose   = num_pose_list[0]
    num_sensor = len(trajectories_list)
    num_solution = num_sensor-1
    print("Input: %d sensors, each has %d poses" % (num_sensor, num_pose))

    '''Assemble observation matrix lm, each row l = [ra,ta,  ..., rm, tm]'''
    stacked_r_list = [ np.vstack( [ Rot2ax(pose_mat[:3,:3]) for pose_mat in trajectory] )
                        for trajectory in trajectories_list ]
    stacked_t_list = [ np.vstack( [ pose_mat[:3, 3]         for pose_mat in trajectory] )
                        for trajectory in trajectories_list ]
    r_t_interleaved = map(np.hstack, zip(stacked_r_list, stacked_t_list))
    lm = np.hstack( r_t_interleaved )   # lm.shape = (num_pose, 6*num_sensor)

    '''Assemble covariance matrix '''
    if trajectories_cov_list is None:
        Cov_ll = np.tile(np.eye(6*num_sensor), (num_pose, 1, 1))
    else:
        Cov_ll = np.zeros((num_pose, 6*num_sensor, 6*num_sensor))
        cov_list_time_majored = zip(*trajectories_cov_list) # list[sensor_idx][pose_idx] -> list[pose_idx][sensor_idx]
        for pose_idx in range(num_pose):
            Cov_ll[pose_idx, :, :] = scipy.linalg.block_diag(*cov_list_time_majored[pose_idx])

    '''Calculate close form solution as initial guess'''
    x0_list = []
    I3 = np.eye(3)
    for s in range(1, num_sensor):
        # rotation first
        H = stacked_r_list[0].T.dot(stacked_r_list[s])
        U, d, Vt = np.linalg.svd(H)
        R = Vt.T.dot(U.T)

        # then translation
        A = np.vstack([ I3 - ax2Rot(r_) for r_ in stacked_r_list[s]])
        b = np.hstack( stacked_t_list[s] - ( R.dot(stacked_t_list[0].T) ).T )
        t = np.linalg.lstsq(A, b)[0]
        x0_list.append([Rot2ax(R), t])
    x0 = np.array(x0_list).flatten()
    print('Initial guess:')
    map(lambda rt: print(MfromRT(*rt)), x0_list)

    '''Assemble constraint functions '''
    def g(x, l):
        x = np.reshape(x, (num_solution, 6))
        l = np.reshape(l, (num_sensor,   6))
        r,t = np.split(l, 2, axis=1)
        e = []
        for x_s, s in zip(x, range(1, num_sensor)):
            Rq = ax2Rot(x_s[0:3])
            Rs = ax2Rot(r[s])
            e.append(x_s[3:] - Rs.dot(x_s[3:]) + Rq.dot(t[0]) - t[s]) # g1 constraints
            e.append( Rq.dot(r[0]) - r[s] )                           # full-constraints
        return np.hstack(e)
    g_diff = GenerateAutoDiffFunction(g, x0, lm[0,:])

    '''solve'''
    xnu, Cov_xx, sigma_0, vv, w = SolveGaussHelmertProblem(g_diff, x0, lm, Cov_ll, *args)
    return [MfromRT(x[:3], x[3:]) for x in np.split(xnu, num_solution) ]

#%%

def demo_and_test():
    def randsp(n=3):
        v = np.random.uniform(-1, 1, size=n)
        return v/np.linalg.norm(v)

    ''' ground truth transformation between sensors '''
    num_sensor = 3
    num_solution = num_sensor-1
    x_true = np.array([randsp() for _ in range(num_solution*2) ]).ravel()   # x= [r1,t1,...,rn,tn]
    Hm = [MfromRT(x[:3], x[3:]) for x in np.split(x_true, num_solution)]

    ''' generate ground truth trajectories '''
    num_pose = 500
    dM = []
    Hm_inv = [np.linalg.inv(h) for h in Hm]
    for t in xrange(num_pose):
        dm = [MfromRT(randsp(),randsp())]   # base sensor
        for h, h_inv in zip(Hm, Hm_inv):    # other sensor
            dm.append( h.dot(dm[0]).dot(h_inv) )
        dM.append(dm)
    trajectories_list = zip(*dM)

    ''' add measurement noise'''
    sigma_r = 1e-3
    sigma_t = 1e-2

    noisy_trajectories = []
    for trajectory in trajectories_list:
        one_trajectory = []
        for pose in trajectory:
            r,t = RTfromM(pose)
            new_pose = MfromRT( r+sigma_r*randn(3), t+sigma_t*randn(3))
            one_trajectory.append(new_pose)
        noisy_trajectories.append(one_trajectory)
    trajectory_covs = [ [np.diag([sigma_r]*3 + [sigma_t]*3)**2] * num_pose ] * num_sensor

    H_est = ExtrinsicCalibration3D(noisy_trajectories, trajectory_covs)
    print("After refinement:")
    map(print, H_est)
    print("Ground truth:")
    map(print, Hm)

if __name__ =='__main__':
   demo_and_test()