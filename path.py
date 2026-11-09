#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau

Part 1.II - motion planning under the grasping constraint.

II.a sampleconfiguration: random cube placement, then inverse geometry.
II.b projectpath: interpolate q and the cube placement, and project every
     interpolated q back onto the grasping constraint.
II.c computepath: RRT-Connect (Motion Planning II lecture; Kuffner & LaValle,
     ICRA 2000) over grasping configurations, with projectpath as the local
     steering method, then random shortcutting.
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import norm
from scipy.spatial import cKDTree

from config import LEFT_HAND, LEFT_HOOK
import time

from tools import setcubeplacement
from inverse_geometry import computeqgrasppose_fast, isvalid

DISCRETISATION = 0.02   # (rad) max step in C between 2 configurations of a path
CUBE_STEP      = 0.01   # (m) max cube displacement between 2 configurations
MAX_JUMP       = 0.15   # (rad) a larger step means the IK jumped to another branch
RRT_DELTA      = 0.10   # (m) max cube displacement in one RRT extension
RRT_MAX_ITER   = 500    # iterations of one RRT-Connect run
N_ATTEMPTS     = 3      # RRT-Connect runs before reporting a failure
N_SHORTCUTS    = 30     # shortcut attempts on the solution
CLEARANCE_Z    = 0.35   # (m) sampling height above the start / goal
MARGIN_XY      = 0.15   # (m) sampling margin around the start / goal
TABLE_Z        = 0.93   # (m) height of the cube resting on the table
PATH_CLEARANCE = 0.02   # (m) clearance from the obstacle along the path,
CLEARANCE_RAMP = 0.15   # (m) reduced to 0 within this distance of the start / goal
ROT_WEIGHT     = 0.1    # (m/rad) weight of the rotation in the cube distance

_rng = np.random.default_rng(42)

# computepath takes no robot / cube argument, so the planner loads its own models
_ROBOT, _CUBE = None, None

def getmodels():
    global _ROBOT, _CUBE
    if _ROBOT is None:
        from setup_pinocchio import setuppinocchio
        _ROBOT, _table, _obstacle, _CUBE = setuppinocchio()
    return _ROBOT, _CUBE

# start and goal cube placements of the current query, set by computepath
_ENDPOINTS = None


def requiredclearance(Mcube):
    '''Clearance (m) required from the obstacle when the cube is at Mcube.
    In pybullet, a path that grazes the obstacle makes the wrist catch it, so
    PATH_CLEARANCE is required; the start and goal poses themselves can be
    closer (the default goal pose is ~2 mm away), so the requirement is ramped
    down near them.'''
    if _ENDPOINTS is None:
        return 0.
    d = min(norm(Mcube.translation - M.translation) for M in _ENDPOINTS)
    return PATH_CLEARANCE * min(1., d / CLEARANCE_RAMP)


def cubeplacementfromq(robot, cube, q):
    '''Cube placement given by the left hand of a grasping configuration:
    oMcube = oMleft * (cubeMhookL)^-1'''
    pin.framesForwardKinematics(robot.model, robot.data, q)
    oML = robot.data.oMf[robot.model.getFrameId(LEFT_HAND)]
    cubeMhookL = cube.data.oMf[cube.model.getFrameId(LEFT_HOOK)]
    return oML * cubeMhookL.inverse()


def interpolatecube(M0, M1, t):
    '''Linear interpolation of the translation, geodesic (exp / log) of the rotation'''
    R = M0.rotation @ pin.exp3(t * pin.log3(M0.rotation.T @ M1.rotation))
    p = (1 - t) * M0.translation + t * M1.translation
    return pin.SE3(R, p)


def cubefeature(M):
    '''Vector used for the nearest neighbour search: [position, ROT_WEIGHT * log3(R)].
    The Euclidean distance between features is the distance between cube placements.'''
    return np.concatenate([M.translation, ROT_WEIGHT * pin.log3(M.rotation)])


def samplingbounds(cubeplacementq0, cubeplacementqgoal):
    '''Box in which cube positions are sampled: the region spanned by the start
    and goal, with a margin, from the table up to CLEARANCE_Z above them.'''
    p0, p1 = cubeplacementq0.translation, cubeplacementqgoal.translation
    lo = np.minimum(p0, p1) - np.array([MARGIN_XY, MARGIN_XY, 0.])
    hi = np.maximum(p0, p1) + np.array([MARGIN_XY, MARGIN_XY, CLEARANCE_Z])
    lo[2] = max(lo[2], TABLE_Z)
    return lo, hi


# II.a sampling configurations
def samplecubeplacement(cubeplacementq0, cubeplacementqgoal):
    '''Position uniform in the sampling box, orientation between the start and
    goal orientations (the cube is never tilted).'''
    lo, hi = samplingbounds(cubeplacementq0, cubeplacementqgoal)
    p = _rng.uniform(lo, hi)
    M = interpolatecube(cubeplacementq0, cubeplacementqgoal, _rng.uniform())
    return pin.SE3(M.rotation, p)


def sampleconfiguration(robot, cube, qseed, cubeplacementq0, cubeplacementqgoal, maxtrials=20):
    '''Random valid grasping configuration: returns (q, cube placement, success)'''
    for _ in range(maxtrials):
        M = samplecubeplacement(cubeplacementq0, cubeplacementqgoal)
        q, ok = computeqgrasppose_fast(robot, qseed, cube, M, requiredclearance(M))
        if ok:
            return q, M, True
    return qseed, None, False


# II.b path projection
def projectpath(robot, cube, q0, q1, cube0=None, cube1=None, discretisationstep=DISCRETISATION):
    '''Path from q0 to q1 in which every configuration grasps the cube.
    Returns (path, success); on failure, path is the valid part from q0.
    Each step interpolates q and the cube placement, then projects the
    interpolated q onto the grasping constraint (IK seeded with it).'''
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


# II.c RRT-Connect
class Tree:
    '''Grasping configurations with their cube placement and parent index.
    Nearest neighbour queries use a k-d tree (scipy cKDTree) on cubefeature,
    rebuilt when nodes have been added since the last query.'''
    def __init__(self, q, Mcube):
        self.q, self.cube, self.parent = [q], [Mcube], [None]
        self.features = [cubefeature(Mcube)]
        self.kdtree, self.kdsize = None, 0

    def add(self, q, Mcube, parent):
        self.q.append(q); self.cube.append(Mcube); self.parent.append(parent)
        self.features.append(cubefeature(Mcube))
        return len(self.q) - 1

    def nearest(self, Mcube):
        if self.kdsize != len(self.features):
            self.kdtree, self.kdsize = cKDTree(np.array(self.features)), len(self.features)
        return int(self.kdtree.query(cubefeature(Mcube))[1])

    def branch(self, idx):
        '''configurations from the root to node idx'''
        res = []
        while idx is not None:
            res.append(self.q[idx]); idx = self.parent[idx]
        return res[::-1]


def steer(Mnear, Mrand, delta=RRT_DELTA):
    '''Cube placement at most delta (m) from Mnear, towards Mrand'''
    d = norm(Mrand.translation - Mnear.translation)
    if d <= delta:
        return Mrand
    return interpolatecube(Mnear, Mrand, delta / d)


def extend(robot, cube, tree, Mtarget):
    '''Extend tree towards Mtarget: returns (new node index or None, Mtarget reached)'''
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
    '''Extend tree towards Mtarget until it is reached or blocked, then link the
    last node to qtarget. Returns the index of that node, or None.'''
    while True:
        inew, reached = extend(robot, cube, tree, Mtarget)
        if inew is None:
            return None
        if reached:
            _, ok = projectpath(robot, cube, tree.q[inew], qtarget, tree.cube[inew], Mtarget)
            return inew if ok else None


def rrtconnect(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal, maxiter=RRT_MAX_ITER):
    '''Node configurations from qinit to qgoal, or None'''
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
    '''Replace the part between 2 random nodes by a direct projected path when valid'''
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
    '''Concatenate the projected paths between consecutive nodes'''
    path = [nodes[0]]
    for qa, qb in zip(nodes[:-1], nodes[1:]):
        seg, ok = projectpath(robot, cube, qa, qb)
        assert ok, "a previously validated edge is no longer valid"
        path += seg[1:]
    return path


#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(qinit,qgoal,cubeplacementq0, cubeplacementqgoal):
    '''If no path is found after N_ATTEMPTS runs, prints an error and returns
    [qinit, qgoal], which is not a valid path.'''
    global _ENDPOINTS
    robot, cube = getmodels()
    _ENDPOINTS = (cubeplacementq0, cubeplacementqgoal)
    t0 = time.time()
    for attempt in range(N_ATTEMPTS):
        nodes = rrtconnect(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal)
        if nodes is not None:
            break
        print("computepath: attempt %d failed, retrying" % (attempt + 1))
    if nodes is None:
        print("computepath: ERROR, no path found, returning [qinit, qgoal]")
        return [qinit, qgoal]
    nodes = shortcut(robot, cube, nodes)
    path = densify(robot, cube, nodes)
    print("computepath: %d configurations, %.1f s" % (len(path), time.time() - t0))
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

