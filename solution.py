#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 11:43:26 2023

@author: stonneau
"""

from os.path import dirname, join, abspath
import numpy as np
import pinocchio as pin #the pinocchio library
from pinocchio.utils import rotate

#helpers
#if needed, you can store the placement of the right hand in the left hand frame here
#
# When the cube is grasped, LEFT_HAND = LEFT_HOOK and RIGHT_HAND = RIGHT_HOOK, so
# the relative placement of the hands is the relative placement of the hooks in
# the cube urdf (models/cubes/cube_small.urdf):
#   LEFT_HOOK  : xyz = (0,  0.05, 0), rpy = (0, 0, 0)
#   RIGHT_HOOK : xyz = (0, -0.05, 0), rpy = (0, 0, -3.14)
# so  LMR = LMcube * cubeMR = (0, -0.1, 0) with a rotation of -3.14 rad around z.
# This was checked numerically from the q0 / qe returned by computeqgrasppose
# (see the check at the bottom of this file).
LMRREF = pin.SE3(rotate('z', -3.14), np.array([0., -0.1, 0.]))
RMLREF = LMRREF.inverse()


if __name__ == "__main__":
    from setup_pinocchio import setuppinocchio
    from inverse_geometry import computeqgrasppose
    from config import CUBE_PLACEMENT, LEFT_HAND, RIGHT_HAND
    robot, table, obstacle, cube = setuppinocchio()
    q0, ok = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT)
    pin.framesForwardKinematics(robot.model, robot.data, q0)
    oML = robot.data.oMf[robot.model.getFrameId(LEFT_HAND)]
    oMR = robot.data.oMf[robot.model.getFrameId(RIGHT_HAND)]
    print("measured LMR:\n", oML.inverse() * oMR)
    print("stored LMRREF:\n", LMRREF)
    print("difference (norm of log):", np.linalg.norm(pin.log((oML.inverse() * oMR).inverse() * LMRREF).vector))
