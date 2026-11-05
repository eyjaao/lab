#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau

Part 1.I - Inverse geometry: compute a configuration where both effectors are
aligned with the cube hooks.

Approach (my own implementation, ideas from the IK lecture and the inverse
geometry part of lab 0 / tutorial 3):
    * Iterative Jacobian-based IK (Gauss-Newton with damping, i.e. damped
      least squares) on the stacked 6D error of the two hands. The 6D error of
      each hand is the SE(3) log of handMhook, as in the 6D IG of lab 0.
    * A postural bias is added in the null space of the hand task so that the
      solution stays close to a natural reference posture (see the hint in the
      instructions). This is the classic "secondary task in the null space"
      construction.
    * The configuration is projected onto the joint limits after every step.
    * Once converged, the solution is checked for collisions and joint limits.
      If it fails, a few random restarts around the reference posture are tried
      before giving up.
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits, jointlimitsviolated
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement

# ---------------------------------------------------------------------------
# Solver parameters
# ---------------------------------------------------------------------------
IK_MAX_ITER   = 200     # max Gauss-Newton iterations per attempt
IK_DT         = 1.0     # step length (full Newton step, the damping keeps it safe)
IK_DAMPING    = 1e-3    # Levenberg-Marquardt damping on J J^T
IK_TOL        = 1e-4    # stopping tolerance on the norm of the 6D+6D error
POSTURE_GAIN  = 0.05    # gain of the postural task (null-space of the hands task)
N_RESTARTS    = 40      # random restarts if the first attempts fail
RESTART_NOISE = 1.0     # amplitude (rad) of the random perturbation of the seeds
N_WIDE_RESTARTS = 200   # last-resort restarts from uniformly random configurations
HEAD_IDX      = [1, 2]  # indices of HEAD_JOINT0 / HEAD_JOINT1 in q


def _posturalreference(robot):
    '''A natural "arms forward, elbows bent" posture used as postural bias.
    Only the chest and arm joints matter; the head is kept at 0.
    Values chosen by hand by looking at the robot in meshcat (they only
    bias the solution, they are never imposed as constraints).'''
    qref = robot.q0.copy()
    # joint order (pinocchio): CHEST, HEAD0, HEAD1, LARM0..5, RARM0..5
    qref[3:9]  = np.array([0.,  -0.3, -1.6, 0., 0., 0.])   # left arm
    qref[9:15] = np.array([0.,  -0.3, -1.6, 0., 0., 0.])   # right arm
    return projecttojointlimits(robot, qref)


def handserror(robot, q, cube):
    '''Return the stacked 12D error between the hands and the hooks (expressed
    in the hand frames) and the stacked 12xnv jacobian (also in the hand
    frames), for the configuration q. The cube placement used is the current
    one (set with setcubeplacement).'''
    pin.framesForwardKinematics(robot.model, robot.data, q)
    pin.computeJointJacobians(robot.model, robot.data, q)

    idL = robot.model.getFrameId(LEFT_HAND)
    idR = robot.model.getFrameId(RIGHT_HAND)
    oML = robot.data.oMf[idL]
    oMR = robot.data.oMf[idR]
    oMhookL = getcubeplacement(cube, LEFT_HOOK)
    oMhookR = getcubeplacement(cube, RIGHT_HOOK)

    # 6D error = log(handMhook), i.e. the twist to apply to the hand (expressed
    # in the hand frame) to reach the hook in unit time.
    eL = pin.log(oML.inverse() * oMhookL).vector
    eR = pin.log(oMR.inverse() * oMhookR).vector

    JL = pin.getFrameJacobian(robot.model, robot.data, idL, pin.LOCAL)
    JR = pin.getFrameJacobian(robot.model, robot.data, idR, pin.LOCAL)
    return np.concatenate([eL, eR]), np.vstack([JL, JR])


def _solveik(robot, qinit, cube, qposture):
    '''Damped least-squares IK with a postural task in the null space.
    Returns the final configuration and the final error norm.'''
    q = qinit.copy()
    nv = robot.model.nv
    err = np.inf
    for _ in range(IK_MAX_ITER):
        e, J = handserror(robot, q, cube)
        err = norm(e)
        if err < IK_TOL:
            break
        # damped pseudo-inverse: J^T (J J^T + lambda I)^-1
        Jpinv = J.T @ inv(J @ J.T + IK_DAMPING * np.eye(J.shape[0]))
        vq = Jpinv @ e
        # postural bias, projected in the null space of the hands task
        P = np.eye(nv) - Jpinv @ J
        vq += P @ (POSTURE_GAIN * (qposture - q))
        q = pin.integrate(robot.model, q, vq * IK_DT)
        q = projecttojointlimits(robot, q)
    return q, err


def obstacleclearance(robot, q):
    '''Smallest distance between the obstacle and the robot / cube at q.
    Uses the collision pairs of robot.collision_model that involve the
    obstacle (the cube placement must already be set).'''
    model, data = robot.collision_model, robot.collision_data
    if not hasattr(robot, '_obstaclepairs'):
        gid = model.getGeometryId('obstaclebase_0')
        robot._obstaclepairs = [k for k, cp in enumerate(model.collisionPairs)
                                if cp.first == gid or cp.second == gid]
    pin.updateGeometryPlacements(robot.model, robot.data, model, data, q)
    return min(pin.computeDistance(model, data, k).min_distance for k in robot._obstaclepairs)


def isvalid(robot, q, margin=0.):
    '''True if q respects the joint limits, is collision free, and keeps at
    least margin (m) between the obstacle and the robot / cube.
    The margin was added after testing in pybullet: with a path that only
    grazes the obstacle, a small tracking error is enough for the wrist to
    catch it. It is used by the planner (path.py) and as a preference in
    computeqgrasppose.'''
    if jointlimitsviolated(robot, q) or collision(robot, q):
        return False
    return margin <= 0. or obstacleclearance(robot, q) >= margin


def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)

    qposture = _posturalreference(robot)

    # The first two attempts use the natural posture as postural bias, starting
    # from the current configuration (useful when called from the planner with
    # a nearby configuration), then from the posture itself.
    # If both fail (in practice: the converged pose is in collision, typically
    # an elbow hitting the chest or the obstacle), random restarts are used.
    # For a restart the postural bias is the random seed itself, otherwise the
    # bias would pull every restart back to the same colliding solution.
    attempts = [(qcurrent.copy(), qposture), (qposture.copy(), qposture)]
    rng = np.random.default_rng(0)  # fixed seed -> reproducible results
    for _ in range(N_RESTARTS):
        noise = rng.uniform(-RESTART_NOISE, RESTART_NOISE, robot.model.nq)
        qseed = projecttojointlimits(robot, qposture + noise)
        attempts.append((qseed, qseed))
    # Last resort, for difficult targets (e.g. cube on the table next to the
    # obstacle): seeds drawn uniformly in the joint limits. The 6D tasks of the
    # two hands use 12 of the 15 dofs, so each arm is almost fully determined
    # by the chest angle and the IK "branch" (elbow up/down, wrist flipped);
    # only seeds far from the posture reach some of the branches.
    for _ in range(N_WIDE_RESTARTS):
        qseed = pin.randomConfiguration(robot.model)
        qseed[HEAD_IDX] = 0.   # the head plays no role, keep it straight
        attempts.append((qseed, qseed))

    # Return the first valid solution. The attempts are ordered from the most
    # natural (posture-biased) to the least, so the first valid one is also
    # the most natural. (I tried preferring solutions far from the obstacle
    # instead, but it picks unusual arm "branches" for q0 and qe, which the
    # planner then cannot connect; the clearance is handled in path.py.)
    qbest, errbest = qcurrent.copy(), np.inf
    for qseed, qbias in attempts:
        q, err = _solveik(robot, qseed, cube, qbias)
        if err < IK_TOL * 10 and isvalid(robot, q):
            if viz is not None:
                from setup_meshcat import updatevisuals
                updatevisuals(viz, robot, cube, q)
            return q, True
        if err < errbest:
            qbest, errbest = q, err
    return qbest, False


def computeqgrasppose_fast(robot, qcurrent, cube, cubetarget, margin=0.):
    '''Single attempt version used inside the planner (path.py): start from
    qcurrent only, no restarts. Much cheaper, and keeps the solution close to
    qcurrent, which is what we want when projecting a path. margin is the
    required clearance from the obstacle (see isvalid).'''
    setcubeplacement(robot, cube, cubetarget)
    q, err = _solveik(robot, qcurrent, cube, _posturalreference(robot))
    return q, (err < IK_TOL * 10 and isvalid(robot, q, margin))


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()

    q = robot.q0.copy()

    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)

    print("q0 success:", successinit, "\nqe success:", successend)
    updatevisuals(viz, robot, cube, q0)



