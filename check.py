#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustness check for computeqgrasppose: random cube placements, success rate.
Not used by the marking code. Run with: python check_ig.py
"""
import numpy as np
import pinocchio as pin
from pinocchio.utils import rotate

from setup_pinocchio import setuppinocchio
from tools import collision, jointlimitsviolated, setcubeplacement
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from inverse_geometry import computeqgrasppose

N_TRIALS = 100
NOISE_XY = 0.05      # metres of noise around the nominal start / target
NOISE_YAW = 0.3      # radians of noise around z
SEED = 0


def noisy(placement, rng):
    p = placement.translation.copy()
    p[:2] += rng.uniform(-NOISE_XY, NOISE_XY, 2)
    R = rotate('z', rng.uniform(-NOISE_YAW, NOISE_YAW)) @ placement.rotation
    return pin.SE3(R, p)


def cube_in_collision(robot, cube, oMc):
    setcubeplacement(robot, cube, oMc)
    return pin.computeCollisions(cube.collision_model, cube.collision_data, False)


if __name__ == "__main__":
    robot, table, obstacle, cube = setuppinocchio()
    rng = np.random.default_rng(SEED)

    for name, nominal in [("start", CUBE_PLACEMENT), ("target", CUBE_PLACEMENT_TARGET)]:
        tried, ok, lied = 0, 0, 0
        for _ in range(N_TRIALS):
            oMc = noisy(nominal, rng)
            if cube_in_collision(robot, cube, oMc):
                continue  # cube itself overlaps the obstacle: skip, not a fair test
            tried += 1
            q, success = computeqgrasppose(robot, robot.q0.copy(), cube, oMc, None)
            really_valid = not collision(robot, q) and not jointlimitsviolated(robot, q)
            ok += bool(success)
            lied += bool(success and not really_valid)
        print(f"{name:6s}: {ok}/{tried} succeeded, {lied} reported success but were invalid")