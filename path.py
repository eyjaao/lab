#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau

Part 1.II - Motion planning under the grasping constraint.

Overview of the approach (RRT and RRT-Connect from the Motion Planning II
lecture; the general RRT structure is inspired by the motion planning tutorial,
but the code below is my own):

II.a  sampleconfiguration : sample a cube placement in a bounded region above
      the table, then solve an inverse geometry problem (inverse_geometry.py)
      to obtain a collision free configuration grasping the cube there.

II.b  projectpath : given two grasping configurations q0, q1 and a step,
      linearly interpolate in configuration space AND in cube space, and
      project every interpolated configuration back onto the grasping
      constraint with the IG solver, seeded with the interpolated
      configuration. Every configuration returned is therefore a grasping
      pose, collision free and within the joint limits. If one of them cannot
      be projected (or the projection "jumps" to another IK branch), the
      function stops and returns the valid part of the path and False.

II.c  computepath : RRT-Connect in the space of grasping configurations
      (bidirectional trees from q0 and qe, greedy connection). The edges are
      the projected paths of II.b. The raw solution is then shortened with a
      few random shortcuts (also validated with projectpath), and finally
      densified so that consecutive configurations are at most DISCRETISATION
      apart.
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv, norm

from config import LEFT_HAND, RIGHT_HAND, LEFT_HOOK
import time

from tools import setcubeplacement, collision, jointlimitsviolated
from inverse_geometry import computeqgrasppose_fast, computeqgrasppose, isvalid

# ---------------------------------------------------------------------------
# Planner parameters
# ---------------------------------------------------------------------------
DISCRETISATION   = 0.02   # max distance (in C, rad) between 2 configs of a projected path
CUBE_STEP        = 0.01   # max distance (m) travelled by the cube between 2 configs
MAX_JUMP         = 0.15   # a projected config further than this from its predecessor = branch jump
RRT_DELTA        = 0.10   # max distance (m) the cube moves in one RRT extension
RRT_MAX_ITER     = 500    # max number of RRT-Connect iterations
N_SHORTCUTS      = 30     # number of shortcut attempts on the solution path
CLEARANCE_Z      = 0.35   # how high above the start/goal the cube may be sampled (m)
MARGIN_XY        = 0.15   # margin around start/goal for the sampled cube positions (m)
TABLE_Z          = 0.93   # height of the cube on the table (CUBE_PLACEMENT z)
PATH_CLEARANCE   = 0.02   # (m) clearance from the obstacle required along the path...
CLEARANCE_RAMP   = 0.15   # ...reduced linearly to 0 when the cube gets within this distance (m) of the start / goal

_rng = np.random.default_rng(42)  # fixed seed so that results are reproducible


# ---------------------------------------------------------------------------
# computepath has no robot / cube argument (its signature must not be changed),
# so the planner keeps its own pinocchio models, loaded once, lazily.
# ---------------------------------------------------------------------------
_ROBOT, _CUBE = None, None

def getmodels():
    global _ROBOT, _CUBE
    if _ROBOT is None:
        from setup_pinocchio import setuppinocchio
        _ROBOT, _table, _obstacle, _CUBE = setuppinocchio()
    return _ROBOT, _CUBE


# start / goal cube placements of the current query (set by computepath),
# used to compute the required clearance
_ENDPOINTS = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def requiredclearance(Mcube):
    '''Clearance (m) required between the obstacle and the robot / cube when
    the cube is at Mcube. Tested in pybullet: a path that grazes the obstacle
    is not trackable (the wrist catches it), so PATH_CLEARANCE is required
    along the path. The start / goal configurations cannot always keep that
    distance (e.g. for the default target an arm has to pass within a few
    mm of the obstacle), so the requirement is ramped down to 0 as the cube
    gets close to its start or goal placement.'''
    if _ENDPOINTS is None:
        return 0.
    d = min(norm(Mcube.translation - M.translation) for M in _ENDPOINTS)
    return PATH_CLEARANCE * min(1., d / CLEARANCE_RAMP)


def cubeplacementfromq(robot, cube, q):
    '''Cube placement implied by the left hand of a grasping configuration q:
    oMcube = oMleft * (cubeMhookL)^-1'''
    pin.framesForwardKinematics(robot.model, robot.data, q)
    oML = robot.data.oMf[robot.model.getFrameId(LEFT_HAND)]
    cubeMhookL = cube.data.oMf[cube.model.getFrameId(LEFT_HOOK)]
    return oML * cubeMhookL.inverse()


def interpolatecube(M0, M1, t):
    '''Interpolation between 2 cube placements: linear for the translation,
    geodesic (slerp, via exp/log) for the rotation.'''
    R = M0.rotation @ pin.exp3(t * pin.log3(M0.rotation.T @ M1.rotation))
    p = (1 - t) * M0.translation + t * M1.translation
    return pin.SE3(R, p)


def cubedistance(M0, M1):
    '''Distance used for the nearest neighbour search: translation distance
    plus a (small) weight on the rotation angle'''
    return norm(M0.translation - M1.translation) + 0.1 * norm(pin.log3(M0.rotation.T @ M1.rotation))


def samplingbounds(cubeplacementq0, cubeplacementqgoal):
    '''Box in which cube positions are sampled. I bound the problem (allowed by
    the instructions) to the region spanned by the start and the goal, with a
    margin, from the table height up to CLEARANCE_Z above the higher of the 2.'''
    p0, p1 = cubeplacementq0.translation, cubeplacementqgoal.translation
    lo = np.minimum(p0, p1) - np.array([MARGIN_XY, MARGIN_XY, 0.])
    hi = np.maximum(p0, p1) + np.array([MARGIN_XY, MARGIN_XY, CLEARANCE_Z])
    lo[2] = max(lo[2], TABLE_Z)
    return lo, hi


# ---------------------------------------------------------------------------
# II.a  Sampling configurations
# ---------------------------------------------------------------------------
def samplecubeplacement(cubeplacementq0, cubeplacementqgoal):
    '''Random cube placement: position uniform in the sampling box, orientation
    on the geodesic between the start and goal orientations (the cube is never
    tilted, which is what we want when carrying it).'''
    lo, hi = samplingbounds(cubeplacementq0, cubeplacementqgoal)
    p = _rng.uniform(lo, hi)
    M = interpolatecube(cubeplacementq0, cubeplacementqgoal, _rng.uniform())
    return pin.SE3(M.rotation, p)


def sampleconfiguration(robot, cube, qseed, cubeplacementq0, cubeplacementqgoal, maxtrials=20):
    '''Sample a random cube placement and return a valid grasping configuration
    for it (collision free, within joint limits), the placement and a flag.'''
    for _ in range(maxtrials):
        M = samplecubeplacement(cubeplacementq0, cubeplacementqgoal)
        q, ok = computeqgrasppose_fast(robot, qseed, cube, M, requiredclearance(M))
        if ok:
            return q, M, True
    return qseed, None, False


# ---------------------------------------------------------------------------
# II.b  Path projection
# ---------------------------------------------------------------------------
def projectpath(robot, cube, q0, q1, cube0=None, cube1=None, discretisationstep=DISCRETISATION):
    '''Interpolated path from q0 to q1 such that every configuration grasps the
    cube. Returns (path, success). path always starts with q0. On failure, path
    is the valid part of the path (from q0 to the last valid configuration).

    Each intermediate configuration is obtained by linearly interpolating q
    AND the cube placement, and then projecting the interpolated q onto the
    grasping constraint (IG seeded with the interpolated q). Seeding with the
    linear interpolation (and not with the previous configuration) keeps the
    projected path close to the straight line and makes it end exactly at q1.'''
    if cube0 is None: cube0 = cubeplacementfromq(robot, cube, q0)
    if cube1 is None: cube1 = cubeplacementfromq(robot, cube, q1)

    nsteps = int(max(np.ceil(norm(q1 - q0) / discretisationstep),
                     np.ceil(norm(cube1.translation - cube0.translation) / CUBE_STEP), 1))
    path = [q0.copy()]
    qprev = q0
    for i in range(1, nsteps + 1):
        t = i / nsteps
        qlin = pin.interpolate(robot.model, q0, q1, t)
        Mcube = interpolatecube(cube0, cube1, t)
        if i == nsteps:
            q = q1.copy()
            setcubeplacement(robot, cube, cube1)
            ok = isvalid(robot, q, requiredclearance(cube1))
        else:
            q, ok = computeqgrasppose_fast(robot, qlin, cube, Mcube, requiredclearance(Mcube))
        if not ok or norm(q - qprev) > MAX_JUMP:
            return path, False
        path.append(q)
        qprev = q
    return path, True


# ---------------------------------------------------------------------------
# II.c  RRT-Connect
# ---------------------------------------------------------------------------
class Tree:
    '''A tree of grasping configurations. Each node stores the configuration,
    the cube placement and the index of its parent.'''
    def __init__(self, q, Mcube):
        self.q, self.cube, self.parent = [q], [Mcube], [None]

    def add(self, q, Mcube, parent):
        self.q.append(q); self.cube.append(Mcube); self.parent.append(parent)
        return len(self.q) - 1

    def nearest(self, Mcube):
        d = [cubedistance(Mcube, M) for M in self.cube]
        return int(np.argmin(d))

    def branch(self, idx):
        '''configurations from the root to node idx'''
        res = []
        while idx is not None:
            res.append(self.q[idx]); idx = self.parent[idx]
        return res[::-1]


def steer(Mnear, Mrand, delta=RRT_DELTA):
    '''Cube placement at most delta (m) away from Mnear, towards Mrand'''
    d = norm(Mrand.translation - Mnear.translation)
    if d <= delta:
        return Mrand
    return interpolatecube(Mnear, Mrand, delta / d)


def extend(robot, cube, tree, Mtarget):
    '''One RRT extension of tree towards the cube placement Mtarget.
    Returns (index of the new node or None, reached Mtarget?)'''
    inear = tree.nearest(Mtarget)
    qnear, Mnear = tree.q[inear], tree.cube[inear]
    Mnew = steer(Mnear, Mtarget)
    qnew, ok = computeqgrasppose_fast(robot, qnear, cube, Mnew, requiredclearance(Mnew))
    if not ok:
        return None, False
    _, ok = projectpath(robot, cube, qnear, qnew, Mnear, Mnew)
    if not ok:
        return None, False
    inew = tree.add(qnew, Mnew, inear)
    return inew, (Mnew is Mtarget)


def connect(robot, cube, tree, Mtarget, qtarget):
    '''Greedy RRT-Connect step: extend tree towards Mtarget until it is reached
    or blocked. When the cube placement is reached, the last node is linked to
    qtarget (the node of the other tree) with a projected path.'''
    while True:
        inew, reached = extend(robot, cube, tree, Mtarget)
        if inew is None:
            return None
        if reached:
            _, ok = projectpath(robot, cube, tree.q[inew], qtarget, tree.cube[inew], Mtarget)
            return inew if ok else None


def rrtconnect(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal, maxiter=RRT_MAX_ITER):
    '''Bidirectional RRT (RRT-Connect, Kuffner & LaValle 2000) over grasping
    configurations. Returns the list of node configurations from qinit to
    qgoal, or None.'''
    # trivial case: the direct projected path is valid
    _, ok = projectpath(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal)
    if ok:
        return [qinit, qgoal]

    ta, tb = Tree(qinit, cubeplacementq0), Tree(qgoal, cubeplacementqgoal)
    a_is_start = True
    for it in range(maxiter):
        qs, Mrand, ok = sampleconfiguration(robot, cube, ta.q[-1], cubeplacementq0, cubeplacementqgoal)
        if not ok:
            continue
        inew, _ = extend(robot, cube, ta, Mrand)
        if inew is not None:
            ilast = connect(robot, cube, tb, ta.cube[inew], ta.q[inew])
            if ilast is not None:
                pa, pb = ta.branch(inew), tb.branch(ilast)
                path = pa + pb[::-1]
                return path if a_is_start else path[::-1]
        ta, tb = tb, ta
        a_is_start = not a_is_start
    return None


def shortcut(robot, cube, nodes, ntrials=N_SHORTCUTS):
    '''Random shortcutting: try to replace the part of the path between 2
    random nodes by a direct projected path. Shortens the (jagged) RRT path.'''
    nodes = list(nodes)
    for _ in range(ntrials):
        if len(nodes) <= 2:
            break
        i, j = sorted(_rng.choice(len(nodes), 2, replace=False))
        if j - i < 2:
            continue
        _, ok = projectpath(robot, cube, nodes[i], nodes[j])
        if ok:
            nodes = nodes[:i + 1] + nodes[j:]
    return nodes


def densify(robot, cube, nodes):
    '''Concatenate the projected paths between consecutive nodes so that the
    final path is a dense list of grasping configurations.'''
    path = [nodes[0]]
    for qa, qb in zip(nodes[:-1], nodes[1:]):
        seg, ok = projectpath(robot, cube, qa, qb)
        assert ok, "a previously validated edge is no longer valid"
        path += seg[1:]
    return path


#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(qinit,qgoal,cubeplacementq0, cubeplacementqgoal):
    global _ENDPOINTS
    robot, cube = getmodels()
    _ENDPOINTS = (cubeplacementq0, cubeplacementqgoal)
    t0 = time.time()
    nodes = rrtconnect(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal)
    if nodes is None:
        print("computepath: no path found, returning [qinit, qgoal]")
        return [qinit, qgoal]
    nodes = shortcut(robot, cube, nodes)
    path = densify(robot, cube, nodes)
    print("computepath: %d nodes, %d configurations, %.1f s" % (len(nodes), len(path), time.time() - t0))
    return path


def displaypath(robot,path,dt,viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose

    robot, cube, viz = setupwithmeshcat()


    q = robot.q0.copy()
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)

    if not(successinit and successend):
        print ("error: invalid initial or end configuration")

    path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    displaypath(robot,path,dt=0.02,viz=viz)

