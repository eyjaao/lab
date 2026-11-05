#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv, inv, norm, svd, eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits, jointlimitsviolated
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

IK_MAX_ITER = 300
IK_STEP = 0.5
IK_DAMPING = 1e-3


def solve_grasp_ik(robot, qinit, oMcubeL, oMcubeR, qref=None):
    '''Damped least-squares IK putting both hands on their hooks.
    Adapted from ARO tutorial 3 (inverse_kinematics_6d_loop).'''
    model, data = robot.model, robot.data
    idL = model.getFrameId(LEFT_HAND)
    idR = model.getFrameId(RIGHT_HAND)
    q = qinit.copy()
    for i in range(IK_MAX_ITER):
        # 1. forward kinematics + jacobians
        pin.framesForwardKinematics(model, data, q)
        pin.computeJointJacobians(model, data, q)

        # 2. current hand placements
        oML = data.oMf[idL]
        oMR = data.oMf[idR]

        # 3. 6D errors, expressed in each hand frame, stacked
        eL = pin.log(oML.inverse() * oMcubeL).vector
        eR = pin.log(oMR.inverse() * oMcubeR).vector
        e = np.hstack([eL, eR])

        # 4. jacobians in the same (LOCAL) frame, stacked
        JL = pin.getFrameJacobian(model, data, idL, pin.LOCAL)
        JR = pin.getFrameJacobian(model, data, idR, pin.LOCAL)
        J = np.vstack([JL, JR])

        # 5. converged?
        if np.linalg.norm(e) < EPSILON:
            return q, True

        # 6. damped least-squares step
        dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING * np.eye(12), e)

        # 7. take the step, stay inside joint limits
        q = pin.integrate(model, q, IK_STEP * dq)
        q = projecttojointlimits(robot, q)

    return q, False


def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)
    oMcubeL = getcubeplacement(cube, LEFT_HOOK)
    oMcubeR = getcubeplacement(cube, RIGHT_HOOK)

    rng = np.random.default_rng()
    q = qcurrent.copy()
    success = False
    for attempt in range(10):
        # a. seed: the given config first, then random perturbations of it
        if attempt == 0:
            seed = qcurrent.copy()
        else:
            seed = projecttojointlimits(robot, qcurrent + rng.uniform(-0.5, 0.5, robot.model.nq))

        # b. solve
        q, conv = solve_grasp_ik(robot, seed, oMcubeL, oMcubeR)

        # c. accept only converged, collision-free, within-limits solutions
        if conv and not collision(robot, q) and not jointlimitsviolated(robot, q):
            success = True
            break

    if viz is not None:
        from setup_meshcat import updatevisuals
        updatevisuals(viz, robot, cube, q)
    return q, success


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()

    q = robot.q0.copy()

    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET, viz)
    print("q0 ok:", successinit, "  qe ok:", successend)

    setcubeplacement(robot, cube, CUBE_PLACEMENT)
    updatevisuals(viz, robot, cube, q0)
    input("enter to see qe")
    setcubeplacement(robot, cube, CUBE_PLACEMENT_TARGET)
    updatevisuals(viz, robot, cube, qe)