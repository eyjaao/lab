#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau

Part 1.I - inverse geometry: a configuration where both effectors are aligned
with the cube hooks.

Iterative IK on the stacked 6D error of the two hands (SE(3) log, as in the 6D
inverse geometry of lab 0), solved with a damped pseudo-inverse of the
Jacobian (IK lecture). A postural task is added in the null space of the hands
task (postural bias hint in the instructions), and q is projected onto the
joint limits after every step.
"""

import pinocchio as pin
import hppfcl
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits, jointlimitsviolated
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement

IK_MAX_ITER     = 200    # max iterations per attempt
IK_DT           = 1.0    # step length
IK_DAMPING      = 1e-3   # damping of the pseudo-inverse
IK_TOL          = 1e-4   # tolerance on the norm of the 12D error
POSTURE_GAIN    = 0.05   # gain of the postural task
N_RESTARTS      = 40     # restarts around the reference posture
RESTART_NOISE   = 1.0    # (rad) amplitude of the perturbation of those restarts
N_WIDE_RESTARTS = 200    # restarts drawn uniformly within the joint limits
HEAD_IDX        = [1, 2] # HEAD_JOINT0 / HEAD_JOINT1 in q


def _posturalreference(robot):
    '''Reference posture for the postural bias: arms forward, elbows bent,
    head straight. Chosen by inspecting the robot in meshcat.'''
    qref = robot.q0.copy()
    # joint order: CHEST, HEAD0, HEAD1, LARM0..5, RARM0..5
    qref[3:9]  = np.array([0., -0.3, -1.6, 0., 0., 0.])
    qref[9:15] = np.array([0., -0.3, -1.6, 0., 0., 0.])
    return projecttojointlimits(robot, qref)


def handserror(robot, q, cube):
    '''12D error log(handMhook) for both hands and the stacked 12 x nv frame
    Jacobian, both expressed in the hand frames (pin.LOCAL).'''
    pin.framesForwardKinematics(robot.model, robot.data, q)
    pin.computeJointJacobians(robot.model, robot.data, q)

    idL = robot.model.getFrameId(LEFT_HAND)
    idR = robot.model.getFrameId(RIGHT_HAND)
    oML = robot.data.oMf[idL]
    oMR = robot.data.oMf[idR]
    oMhookL = getcubeplacement(cube, LEFT_HOOK)
    oMhookR = getcubeplacement(cube, RIGHT_HOOK)

    eL = pin.log(oML.inverse() * oMhookL).vector
    eR = pin.log(oMR.inverse() * oMhookR).vector

    JL = pin.getFrameJacobian(robot.model, robot.data, idL, pin.LOCAL)
    JR = pin.getFrameJacobian(robot.model, robot.data, idR, pin.LOCAL)
    return np.concatenate([eL, eR]), np.vstack([JL, JR])


def _solveik(robot, qinit, cube, qposture):
    '''Damped least-squares IK with a postural task in the null space.
    Returns the final configuration and error norm.'''
    q = qinit.copy()
    nv = robot.model.nv
    err = np.inf
    for _ in range(IK_MAX_ITER):
        e, J = handserror(robot, q, cube)
        err = norm(e)
        if err < IK_TOL:
            break
        Jpinv = J.T @ inv(J @ J.T + IK_DAMPING * np.eye(J.shape[0]))
        vq = Jpinv @ e
        P = np.eye(nv) - Jpinv @ J          # null-space projector
        vq += P @ (POSTURE_GAIN * (qposture - q))
        q = pin.integrate(robot.model, q, vq * IK_DT)
        q = projecttojointlimits(robot, q)
    return q, err


def obstacleclearance(robot, q):
    '''Smallest distance between the obstacle and the robot or cube at q
    (the cube placement must already be set).'''
    model, data = robot.collision_model, robot.collision_data
    if not hasattr(robot, '_obstaclepairs'):
        gid = model.getGeometryId('obstaclebase_0')
        robot._obstaclepairs = [k for k, cp in enumerate(model.collisionPairs)
                                if cp.first == gid or cp.second == gid]
    pin.updateGeometryPlacements(robot.model, robot.data, model, data, q)
    return min(pin.computeDistance(model, data, k).min_distance for k in robot._obstaclepairs)


def isvalid(robot, q, margin=0.):
    '''True if q is within the joint limits, collision free, and at least
    margin (m) away from the obstacle.'''
    if jointlimitsviolated(robot, q) or collision(robot, q):
        return False
    return margin <= 0. or obstacleclearance(robot, q) >= margin


def wristsclear(robot, cube):
    '''False if a grasp at the current cube placement is impossible whatever q:
    the last wrist link is rigidly attached to the hand, so its placement is
    fixed by the hook, oMjoint5 = oMhook * (joint5Mhand)^-1. Its geometry is
    tested against the table and obstacle with hppfcl (used by pinocchio).'''
    m, gmodel = robot.model, robot.collision_model
    env = [gmodel.geometryObjects[gmodel.getGeometryId(n)] for n in ('obstaclebase_0', 'baseLink_0')]
    for side, hook, hand in (('LARM', LEFT_HOOK, LEFT_HAND), ('RARM', RIGHT_HOOK, RIGHT_HAND)):
        jid = m.getJointId(side + '_JOINT5')
        oMj = getcubeplacement(cube, hook) * m.frames[m.getFrameId(hand)].placement.inverse()
        for g in gmodel.geometryObjects:
            if g.parentJoint != jid:
                continue
            oMg = oMj * g.placement
            for e in env:
                res = hppfcl.CollisionResult()
                if hppfcl.collide(g.geometry, hppfcl.Transform3f(oMg.rotation, oMg.translation),
                                  e.geometry, hppfcl.Transform3f(e.placement.rotation, e.placement.translation),
                                  hppfcl.CollisionRequest(), res):
                    return False
    return True


def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)
    if not wristsclear(robot, cube):
        return qcurrent.copy(), False

    qposture = _posturalreference(robot)

    # Seeds, from the most natural to the least: the current configuration and
    # the reference posture (biased towards the posture), then restarts around
    # the posture, then restarts anywhere within the joint limits, needed for
    # targets near the obstacle. A restart is biased towards its own seed so
    # that the postural task does not pull it back to the same solution.
    attempts = [(qcurrent.copy(), qposture), (qposture.copy(), qposture)]
    rng = np.random.default_rng(0)
    lower, upper = robot.model.lowerPositionLimit, robot.model.upperPositionLimit
    for _ in range(N_RESTARTS):
        noise = rng.uniform(-RESTART_NOISE, RESTART_NOISE, robot.model.nq)
        qseed = projecttojointlimits(robot, qposture + noise)
        qseed[HEAD_IDX] = 0.
        attempts.append((qseed, qseed))
    for _ in range(N_WIDE_RESTARTS):
        qseed = rng.uniform(lower, upper)
        qseed[HEAD_IDX] = 0.
        attempts.append((qseed, qseed))

    # first valid solution = the most natural one
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
    '''Single attempt from qcurrent, without restarts, used by the planner.
    margin: required clearance from the obstacle (m).'''
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



