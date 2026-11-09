#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2.I - from a path to a trajectory q(t), vq(t), vvq(t).

level 0: quintic time scaling s(t) = 10u^3 - 15u^4 + 6u^5 (u = t/T) along the
         arc length of the path: zero velocity and acceleration at both ends.
level 1: least-squares fit of a Bezier curve to the path, solved as a QP with
         quadprog (method from the instructions and the ndcurves tutorial):
         - variables: the control points, one QP per joint
         - P0 = P1 = P2 = q_start and the last 3 = q_end: zero velocity and
           acceleration at both ends
         - control points within the joint limits: by the convex hull property
           the whole curve is within the joint limits
         The duration T scales the curve so the peak velocity is VMAX_RATIO of
         the joint velocity limits.
"""

import numpy as np
import pinocchio as pin
from math import comb
import quadprog
from bezier import Bezier

VMAX_RATIO = 0.5    # fraction of the joint velocity limits used
N_CTRL     = 50     # control points of the Bezier curve
SMOOTH_W   = 1e-3   # weight of the smoothness term


def arclength(path):
    '''normalised cumulative arc length of the path, in [0, 1]'''
    path = np.asarray(path)
    d = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    return d / d[-1] if d[-1] > 0 else np.linspace(0., 1., len(path))


def bernsteinmatrix(u, n):
    '''A[i, k] = B_k^n(u_i): the curve at the u_i is A @ P'''
    u = np.asarray(u)[:, None]
    k = np.arange(n + 1)[None, :]
    binom = np.array([comb(n, kk) for kk in range(n + 1)])[None, :]
    return binom * u ** k * (1 - u) ** (n - k)


class Trajectory:
    '''Evaluates f(t) with t clamped to [0, T]'''
    def __init__(self, f, T):
        self.f, self.T = f, T

    def __call__(self, t):
        return self.f(min(max(t, 0.), self.T))


def maketraj_level0(path, T=None, vmax=1.):
    '''Quintic time scaling along the piecewise linear path. If T is None, the
    peak joint velocity is VMAX_RATIO * vmax (the peak of ds/dt is 1.875 / T).'''
    path = np.asarray(path)
    s_nodes = arclength(path)
    seglen = np.diff(s_nodes)
    dqds = np.diff(path, axis=0) / np.where(seglen > 0, seglen, 1.)[:, None]
    if T is None:
        T = 1.875 * np.max(np.abs(dqds)) / (VMAX_RATIO * vmax)

    def s_of_t(t):
        u = t / T
        s   = 10 * u**3 - 15 * u**4 + 6 * u**5
        ds  = (30 * u**2 - 60 * u**3 + 30 * u**4) / T
        dds = (60 * u - 180 * u**2 + 120 * u**3) / T**2
        return s, ds, dds

    def seg(s):
        return int(np.clip(np.searchsorted(s_nodes, s, side='right') - 1, 0, len(path) - 2))

    def q(t):
        s, _, _ = s_of_t(t); i = seg(s)
        a = (s - s_nodes[i]) / seglen[i] if seglen[i] > 0 else 0.
        return (1 - a) * path[i] + a * path[i + 1]

    def vq(t):
        s, ds, _ = s_of_t(t)
        return dqds[seg(s)] * ds

    def vvq(t):
        # dq/ds is constant on each segment of the (dense) path
        s, _, dds = s_of_t(t)
        return dqds[seg(s)] * dds

    return Trajectory(q, T), Trajectory(vq, T), Trajectory(vvq, T), T


def fitbezier(path, ncontrol, qlower, qupper):
    '''Control points minimising sum_i ||B(u_i) - path_i||^2 + SMOOTH_W ||D P||^2,
    where u_i is the normalised arc length and D the 2nd difference of the
    control points (without it the high degree fit oscillates).'''
    path = np.asarray(path)
    nq = path.shape[1]
    n = ncontrol - 1
    A = bernsteinmatrix(arclength(path), n)
    D = np.diff(np.eye(ncontrol), 2, axis=0)
    G = A.T @ A + SMOOTH_W * D.T @ D + 1e-9 * np.eye(ncontrol)  # positive definite

    Ceq = np.zeros((6, ncontrol))
    for r, k in enumerate([0, 1, 2, n - 2, n - 1, n]):
        Ceq[r, k] = 1.
    Cin = np.vstack([np.eye(ncontrol), -np.eye(ncontrol)])  # quadprog: C^T x >= b
    C = np.vstack([Ceq, Cin]).T

    P = np.zeros((ncontrol, nq))
    for j in range(nq):
        b = np.concatenate([[path[0, j]] * 3 + [path[-1, j]] * 3,
                            qlower[j] * np.ones(ncontrol), -qupper[j] * np.ones(ncontrol)])
        P[:, j] = quadprog.solve_qp(G, A.T @ path[:, j], C, b, meq=6)[0]
    return [P[k] for k in range(ncontrol)]


def maketraj_level1(robot, path, ncontrol=N_CTRL):
    '''Bezier trajectory from fitbezier. T is set from the peak velocity of the
    curve in normalised time (the convex hull bound n max|P_k+1 - P_k| / vmax
    is valid but about twice as conservative). Returns q, vq, vvq, T.'''
    path = np.asarray(path)
    ncontrol = min(ncontrol, max(len(path), 7))
    cps = fitbezier(path, ncontrol, robot.model.lowerPositionLimit, robot.model.upperPositionLimit)
    vmax = robot.model.velocityLimit * VMAX_RATIO
    dB = Bezier(cps, t_max=1.).derivative(1)
    peak = np.max([np.abs(dB(u)) for u in np.linspace(0., 1., 2001)], axis=0)
    T = max(float(np.max(peak / vmax)), 1.)
    q_of_t = Bezier(cps, t_max=T)
    vq_of_t = q_of_t.derivative(1)
    vvq_of_t = vq_of_t.derivative(1)
    return Trajectory(q_of_t, T), Trajectory(vq_of_t, T), Trajectory(vvq_of_t, T), T


def envcollisionpairs(robot):
    '''collision pairs involving the table or the obstacle'''
    g = robot.collision_model.geometryObjects
    env = ('obstaclebase_0', 'baseLink_0')
    return [k for k, cp in enumerate(robot.collision_model.collisionPairs)
            if g[cp.first].name in env or g[cp.second].name in env]


def checktrajectory(robot, cube, traj, T, dt=0.01):
    '''Samples where the robot or cube hits the table or obstacle, samples
    outside the joint limits. Hand-cube contacts are not counted: the fitted
    curve satisfies the grasp only approximately.'''
    from tools import jointlimitsviolated, setcubeplacement
    from path import cubeplacementfromq
    pairs = envcollisionpairs(robot)
    ncoll, njl = 0, 0
    for t in np.arange(0., T + dt, dt):
        q = traj[0](t)
        setcubeplacement(robot, cube, cubeplacementfromq(robot, cube, q))
        pin.updateGeometryPlacements(robot.model, robot.data, robot.collision_model, robot.collision_data, q)
        pin.computeCollisions(robot.collision_model, robot.collision_data, False)
        ncoll += any(robot.collision_data.collisionResults[k].isCollision() for k in pairs)
        njl += jointlimitsviolated(robot, q)
    return ncoll, njl


def maketraj(robot, cube, path):
    '''Level 1 if the fitted curve is collision free (more control points bring
    it closer to the path), otherwise level 0, which follows the path exactly.'''
    for ncontrol in (N_CTRL, 60, 80):
        *traj, T = maketraj_level1(robot, path, ncontrol)
        ncoll, njl = checktrajectory(robot, cube, traj, T)
        if ncoll == 0 and njl == 0:
            print("maketraj: level 1, %d control points, T = %.1f s" % (ncontrol, T))
            return (*traj, T)
    *traj, T = maketraj_level0(path, vmax=float(np.min(robot.model.velocityLimit)))
    print("maketraj: level 0, T = %.1f s" % T)
    return (*traj, T)
