#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustness test: runs the full pipeline (inverse geometry, path, trajectory,
control in pybullet without GUI) for start and goal cube placements perturbed
by random noise. A run succeeds if the cube is lifted off the table and ends
within SUCCESS_DIST of the goal. Targets where a grasp is impossible (the
wrist link fixed to the hand hits the obstacle, see wristsclear) are counted
separately.

usage: python test_robustness.py [number of runs]
"""

import sys
import numpy as np
import pinocchio as pin
from pinocchio.utils import rotate

import config
config.USE_PYBULLET_GUI = False

from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, DT
from tools import setupwithpybullet, setcubeplacement
from inverse_geometry import computeqgrasppose, wristsclear
from path import computepath
from trajectory import maketraj
from control import controllaw

NOISE_XY     = 0.03   # (m) uniform noise on the x / y position of start and goal
NOISE_YAW    = 0.2    # (rad) uniform noise on the orientation about z
SUCCESS_DIST = 0.02   # (m)
LIFT_Z       = 0.05   # (m) minimum height reached above the table


def perturb(M, rng):
    p = M.translation + np.array([rng.uniform(-NOISE_XY, NOISE_XY), rng.uniform(-NOISE_XY, NOISE_XY), 0.])
    return pin.SE3(M.rotation @ rotate('z', rng.uniform(-NOISE_YAW, NOISE_YAW)), p)


def run(start, goal):
    '''Full pipeline for one start / goal pair: returns (result, message) with
    result in "success", "fail", "impossible". The simulation is closed by
    Simulation.__del__ when the function returns.'''
    robot, sim, cube = setupwithpybullet()
    p = sim.p
    for M in (start, goal):
        setcubeplacement(robot, cube, M)
        if not wristsclear(robot, cube):
            return "impossible", "wrist link would hit the obstacle"
    q0, ok0 = computeqgrasppose(robot, robot.q0, cube, start)
    qe, oke = computeqgrasppose(robot, robot.q0, cube, goal)
    if not (ok0 and oke):
        return "fail", "inverse geometry failed"
    path = computepath(q0, qe, start, goal)
    if len(path) < 3:
        return "fail", "no path"
    trajs = maketraj(robot, cube, path)
    p.resetBasePositionAndOrientation(sim.cubeId, list(start.translation),
                                      list(pin.Quaternion(start.rotation).coeffs()))
    sim.setqsim(q0)
    t, zmax = 0., 0.
    while t < trajs[3] + 1.:
        controllaw(sim, robot, trajs[:3], t, cube)
        zmax = max(zmax, p.getBasePositionAndOrientation(sim.cubeId)[0][2])
        t += DT
    pos = np.array(p.getBasePositionAndOrientation(sim.cubeId)[0])
    dist = np.linalg.norm(pos[:2] - goal.translation[:2])
    lifted = zmax - start.translation[2]
    ok = dist < SUCCESS_DIST and lifted > LIFT_Z
    return ("success" if ok else "fail"), "final error %.3f m, lifted %.2f m" % (dist, lifted)


if __name__ == "__main__":
    nruns = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    rng = np.random.default_rng(1)
    results = []
    for i in range(nruns):
        start, goal = perturb(CUBE_PLACEMENT, rng), perturb(CUBE_PLACEMENT_TARGET, rng)
        result, msg = run(start, goal)
        results.append(result)
        print("run %d: start %s, goal %s -> %s (%s)" % (i, np.round(start.translation[:2], 3),
              np.round(goal.translation[:2], 3), result, msg))
    nposs = nruns - results.count("impossible")
    print("%d / %d successful (%d impossible targets excluded)"
          % (results.count("success"), nposs, nruns - nposs))
