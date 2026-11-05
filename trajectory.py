#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2.I - From a path to a trajectory.

Two methods are implemented, both returning the triplet (q(t), vq(t), vvq(t))
expected by control.py:

level 0 - maketraj_level0(path, T):
    the path is parametrised by its cumulative arc length s in [0, 1], and s is
    driven by a quintic time scaling s(t) = 10u^3 - 15u^4 + 6u^5 (u = t/T),
    which has zero velocity AND zero acceleration at t = 0 and t = T.
    q(t) is the piecewise linear interpolation of the path at s(t).

level 1 - maketraj_level1(robot, path):
    least-squares fit of a Bezier curve to the path, solved as a quadratic
    program (quadprog), following the method of the lecture / the ndcurves
    tutorial:
      * variables: the control points P_0..P_n of the Bezier curve (one QP per
        joint: the problem is separable because neither the cost nor the
        constraints couple the joints)
      * cost: sum_i || B(u_i) - path_i ||^2 where u_i is the normalised arc
        length of path_i, and B(u_i) is linear in the control points
        (Bernstein polynomials)
      * equality constraints: P_0 = P_1 = P_2 = q_start and
        P_{n-2} = P_{n-1} = P_n = q_end  ->  zero velocity and acceleration at
        both ends, and the curve starts / ends exactly at q0 / qe
      * inequality constraints: lower limit <= P_k <= upper limit. A Bezier
        curve lies in the convex hull of its control points, so this
        guarantees the whole trajectory respects the joint limits.
    Time parametrisation: the fit is done in normalised time u in [0, 1]. The
    derivative of a Bezier curve is a Bezier curve with control points
    n (P_{k+1} - P_k) / T, so the velocity bound |vq| <= VMAX is satisfied over
    the whole trajectory (convex hull again) as soon as
        T >= n max_k |P_{k+1} - P_k| / VMAX
    which gives the duration T directly, without making the problem non-linear.
"""

import numpy as np
import pinocchio as pin
from bezier import Bezier

try:
    import quadprog
except ImportError:  # level 1 needs quadprog (listed in requirements.txt)
    quadprog = None

VMAX_RATIO = 0.5   # use at most this fraction of the joint velocity limits
N_CTRL     = 50    # number of control points of the Bezier curve (level 1)
SMOOTH_W   = 1e-3  # weight of the smoothness (control point 2nd difference) term
TIGHT_T    = True  # duration from the sampled peak velocity (see maketraj_level1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def arclength(path):
    '''normalised cumulative arc length of the path, in [0, 1]'''
    path = np.asarray(path)
    d = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    return d / d[-1] if d[-1] > 0 else np.linspace(0., 1., len(path))


def bernsteinmatrix(u, n):
    '''A[i, k] = B_k^n(u_i) so that the curve evaluated at the u_i is A @ P'''
    from math import comb
    u = np.asarray(u)[:, None]
    k = np.arange(n + 1)[None, :]
    binom = np.array([comb(n, kk) for kk in range(n + 1)])[None, :]
    return binom * u ** k * (1 - u) ** (n - k)


class Trajectory:
    '''Callable wrapper so that q(t), vq(t), vvq(t) can be evaluated for any
    t (clamped to [0, T]: before 0 we return the start, after T the end, with
    zero velocity / acceleration).'''
    def __init__(self, f, T):
        self.f, self.T = f, T

    def __call__(self, t):
        return self.f(min(max(t, 0.), self.T))


# ---------------------------------------------------------------------------
# level 0
# ---------------------------------------------------------------------------
def maketraj_level0(path, T=None, vmax=1.):
    '''Quintic time scaling along the piecewise linear path.
    If T is None it is chosen so that the peak joint velocity is
    VMAX_RATIO * vmax (the peak of ds/dt for the quintic is 1.875 / T).'''
    path = np.asarray(path)
    s_nodes = arclength(path)
    seglen = np.diff(s_nodes)
    # dq/ds on each segment
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
        i = int(np.clip(np.searchsorted(s_nodes, s, side='right') - 1, 0, len(path) - 2))
        return i

    def q(t):
        s, _, _ = s_of_t(t); i = seg(s)
        a = (s - s_nodes[i]) / seglen[i] if seglen[i] > 0 else 0.
        return (1 - a) * path[i] + a * path[i + 1]

    def vq(t):
        s, ds, _ = s_of_t(t)
        return dqds[seg(s)] * ds

    def vvq(t):
        # piecewise linear path: dq/ds is constant on each segment, so only
        # the s'' term remains (the jumps of dq/ds between segments are small
        # since the path is dense).
        s, _, dds = s_of_t(t)
        return dqds[seg(s)] * dds

    return Trajectory(q, T), Trajectory(vq, T), Trajectory(vvq, T), T


# ---------------------------------------------------------------------------
# level 1
# ---------------------------------------------------------------------------
def fitbezier(path, ncontrol, qlower, qupper):
    '''QP fit of the control points (see module doc). Returns a list of
    control points (each of dimension nq).'''
    assert quadprog is not None, "quadprog is required for the level 1 trajectory"
    path = np.asarray(path)
    nq = path.shape[1]
    n = ncontrol - 1                      # degree
    u = arclength(path)
    A = bernsteinmatrix(u, n)             # (npoints, ncontrol)
    # smoothness regulariser: squared 2nd differences of the control points
    # (proportional to the control points of the acceleration curve). Without
    # it the high degree fit oscillates (control points far from the path),
    # which makes the trajectory slow and deviate from the path.
    D = np.diff(np.eye(ncontrol), 2, axis=0)
    G = A.T @ A + SMOOTH_W * D.T @ D + 1e-9 * np.eye(ncontrol) # quadprog needs G positive definite

    # equality constraints on the 3 first / 3 last control points
    Ceq = np.zeros((6, ncontrol))
    for r, k in enumerate([0, 1, 2, n - 2, n - 1, n]):
        Ceq[r, k] = 1.
    # inequality constraints: qlower <= P_k <= qupper  (quadprog: C^T x >= b)
    Cin = np.vstack([np.eye(ncontrol), -np.eye(ncontrol)])

    P = np.zeros((ncontrol, nq))
    for j in range(nq):
        a = A.T @ path[:, j]
        beq = np.array([path[0, j]] * 3 + [path[-1, j]] * 3)
        bin_ = np.concatenate([qlower[j] * np.ones(ncontrol), -qupper[j] * np.ones(ncontrol)])
        C = np.vstack([Ceq, Cin]).T
        b = np.concatenate([beq, bin_])
        P[:, j] = quadprog.solve_qp(G, a, C, b, meq=6)[0]
    return [P[k] for k in range(ncontrol)]


def maketraj_level1(robot, path, ncontrol=N_CTRL):
    '''Bezier trajectory fitted with a QP, duration chosen from the velocity
    limits. Returns q(t), vq(t), vvq(t) (Bezier objects wrapped to clamp t) and T.'''
    path = np.asarray(path)
    ncontrol = min(ncontrol, max(len(path), 7))
    cps = fitbezier(path, ncontrol, robot.model.lowerPositionLimit, robot.model.upperPositionLimit)
    n = ncontrol - 1
    vmax = robot.model.velocityLimit * VMAX_RATIO
    # Duration from the velocity limits. The convex hull bound
    #   T >= n max_k |P_{k+1} - P_k| / vmax
    # guarantees the limit everywhere but is very conservative for a high
    # degree curve, so T is computed from the true peak velocity of the curve
    # in normalised time (sampled finely); the convex hull bound is kept as
    # the guaranteed alternative (TIGHT_T = False).
    if TIGHT_T:
        dB = Bezier(cps, t_max=1.).derivative(1)
        peak = np.max([np.abs(dB(u)) for u in np.linspace(0., 1., 2001)], axis=0)
        T = float(np.max(peak / vmax))
    else:
        dP = np.abs(np.diff(np.array(cps), axis=0))
        T = float(np.max(n * dP / vmax[None, :]))
    T = max(T, 1.)
    q_of_t = Bezier(cps, t_max=T)
    vq_of_t = q_of_t.derivative(1)
    vvq_of_t = vq_of_t.derivative(1)
    wrap = lambda b: Trajectory(b, T)
    return wrap(q_of_t), wrap(vq_of_t), wrap(vvq_of_t), T


def envcollisionpairs(robot):
    '''indices of the collision pairs involving the table or the obstacle'''
    g = robot.collision_model.geometryObjects
    env = ('obstaclebase_0', 'baseLink_0')
    return [k for k, cp in enumerate(robot.collision_model.collisionPairs)
            if g[cp.first].name in env or g[cp.second].name in env]


def checktrajectory(robot, cube, traj, T, dt=0.01):
    '''Diagnostic along the trajectory (the cube is attached to the left hand):
    number of samples where the robot or the cube hits the table / obstacle,
    number of samples violating the joint limits, and the max deviation of the
    relative placement of the hands from the grasp (LMRREF).
    Contacts between the hands and the cube are not counted: the fitted
    trajectory only approximately satisfies the grasp constraint, so the
    hands slightly penetrate the cube, which the force control deals with.'''
    from tools import jointlimitsviolated, setcubeplacement
    from path import cubeplacementfromq
    from config import LEFT_HAND, RIGHT_HAND
    from solution import LMRREF
    q_of_t = traj[0]
    pairs = envcollisionpairs(robot)
    ncoll, njl, maxgrasp = 0, 0, 0.
    for t in np.arange(0., T + dt, dt):
        q = q_of_t(t)
        setcubeplacement(robot, cube, cubeplacementfromq(robot, cube, q))
        pin.updateGeometryPlacements(robot.model, robot.data, robot.collision_model, robot.collision_data, q)
        pin.computeCollisions(robot.collision_model, robot.collision_data, False)
        ncoll += any(robot.collision_data.collisionResults[k].isCollision() for k in pairs)
        njl += jointlimitsviolated(robot, q)
        pin.framesForwardKinematics(robot.model, robot.data, q)
        oML = robot.data.oMf[robot.model.getFrameId(LEFT_HAND)]
        oMR = robot.data.oMf[robot.model.getFrameId(RIGHT_HAND)]
        maxgrasp = max(maxgrasp, np.linalg.norm(pin.log((oML.inverse() * oMR).inverse() * LMRREF).vector))
    return ncoll, njl, maxgrasp


def maketraj(robot, cube, path, verbose=True):
    '''Trajectory used by control.py: level 1 (QP Bezier fit) if the fitted
    trajectory does not hit the table / obstacle, increasing the number of
    control points if needed (more control points = closer to the path);
    falls back to level 0 (which follows the collision free path exactly).
    Returns q(t), vq(t), vvq(t), T.'''
    for ncontrol in (N_CTRL, 60, 80):
        try:
            *traj, T = maketraj_level1(robot, path, ncontrol)
        except Exception as e:  # e.g. quadprog missing
            if verbose: print("maketraj: level 1 failed (%s)" % e)
            break
        ncoll, njl, maxgrasp = checktrajectory(robot, cube, traj, T)
        if verbose:
            print("maketraj: level 1, %d control points, T = %.2f s, collisions: %d, grasp error: %.4f"
                  % (ncontrol, T, ncoll, maxgrasp))
        if ncoll == 0 and njl == 0:
            return (*traj, T)
    *traj, T = maketraj_level0(path, vmax=float(np.min(robot.model.velocityLimit)))
    if verbose: print("maketraj: using level 0, T = %.2f s" % T)
    return (*traj, T)


if __name__ == "__main__":
    from path import getmodels, computepath
    from inverse_geometry import computeqgrasppose
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    robot, cube = getmodels()
    q0, _ = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT)
    qe, _ = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET)
    path = computepath(q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    for name, maker in [("level 0", lambda: maketraj_level0(path, vmax=float(np.min(robot.model.velocityLimit)))),
                        ("level 1", lambda: maketraj_level1(robot, path))]:
        *traj, T = maker()
        ncoll, njl, maxgrasp = checktrajectory(robot, cube, traj, T)
        print("%s: T = %.2f s, samples in collision: %d, joint limits violated: %d, max grasp error: %.4f"
              % (name, T, ncoll, njl, maxgrasp))
