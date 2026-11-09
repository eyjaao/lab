#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau

Part 2.II - torque control.

Inverse dynamics control (pin.crba, pin.nle, hint in the instructions) tracking
the reference trajectory, plus a squeezing force applied on the cube by both
hands through the hand Jacobians, as recommended in the instructions:

    aq_des = aq_ref + Kp (q_ref - q) + Kv (vq_ref - vq)
    tau    = M(q) aq_des + h(q, vq) + J_L^T f_L + J_R^T f_R

The squeezing force is regulated with feedback on the contact normal forces
measured in pybullet (getContactPoints): integral action on the error between
the desired and the measured force.
"""

import numpy as np
import pinocchio as pin

from bezier import Bezier
from config import LEFT_HAND, RIGHT_HAND, DT
    
# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 300.               # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)   # derivative gain (D of PD)

GRASP_FORCE = 30.    # (N) desired squeezing force per hand
GRASP_RAMP  = 0.3    # (s) ramp-up time of the grasp
FORCE_KI    = 20.    # (1/s) integral gain of the force feedback
FORCE_MAX   = 60.    # (N) bounds of the commanded squeezing force
RECORD_VIDEO = False # save the pybullet GUI to demo.mp4 (needs ffmpeg)


def matchsimulatormodel(robot):
    '''The effector links (LARM_EFF_Link, RARM_EFF_Link) have no <inertial> tag
    in the urdf: pinocchio gives them no mass, pybullet a 1 kg point mass
    (pybullet warning at load time, checked with getDynamicsInfo). The point
    masses are added to the pinocchio model, otherwise the arms sag by ~0.2 rad.'''
    if getattr(robot, '_simmodelmatched', False):
        return
    m = robot.model
    for name in (LEFT_HAND, RIGHT_HAND):
        frame = m.frames[m.getFrameId(name)]
        m.inertias[frame.parent] += frame.placement.act(pin.Inertia(1., np.zeros(3), np.zeros((3, 3))))
    robot.data = m.createData()
    robot._simmodelmatched = True


def cubemass(sim):
    '''Mass of the cube in pybullet. Its hook links have no <inertial> tag
    either and get 1 kg each: 2.14 kg in total.'''
    if not hasattr(sim, '_cubemass'):
        p = sim.p
        sim._cubemass = p.getDynamicsInfo(sim.cubeId, -1)[0] + sum(
            p.getDynamicsInfo(sim.cubeId, i)[0] for i in range(p.getNumJoints(sim.cubeId)))
    return sim._cubemass


def measuredsqueeze(sim):
    '''Mean over both hands of the contact normal force between the hand and
    the cube (N), from the last simulation step.'''
    p = sim.p
    if not hasattr(sim, '_armlinks'):
        name = lambda i: p.getJointInfo(sim.robot, i)[12].decode()
        sim._armlinks = {side: {i for i in range(p.getNumJoints(sim.robot)) if name(i).startswith(side)}
                         for side in ('LARM', 'RARM')}
    contacts = p.getContactPoints(sim.robot, sim.cubeId)
    forces = [sum(c[9] for c in contacts if c[3] in sim._armlinks[side]) for side in ('LARM', 'RARM')]
    return 0.5 * sum(forces)


def squeezecommand(sim, fdes):
    '''Commanded squeezing force: fdes + FORCE_KI * integral(fdes - measured)'''
    sim._forceintegral = getattr(sim, '_forceintegral', 0.) + (fdes - measuredsqueeze(sim)) * DT
    sim._forceintegral = float(np.clip(sim._forceintegral, -fdes / FORCE_KI, FORCE_MAX / FORCE_KI))
    return float(np.clip(fdes + FORCE_KI * sim._forceintegral, 0., FORCE_MAX))


def graspforcetorques(robot, q, force, load=0.):
    '''J_L^T f_L + J_R^T f_R: a squeeze of magnitude force along the line
    between the hands, plus an upward force load / 2 on each hand compensating
    the weight of the cube, which is not in the robot model.'''
    pin.computeJointJacobians(robot.model, robot.data, q)
    pin.framesForwardKinematics(robot.model, robot.data, q)
    idL = robot.model.getFrameId(LEFT_HAND)
    idR = robot.model.getFrameId(RIGHT_HAND)
    u = robot.data.oMf[idR].translation - robot.data.oMf[idL].translation
    u = u / np.linalg.norm(u)
    JL = pin.getFrameJacobian(robot.model, robot.data, idL, pin.LOCAL_WORLD_ALIGNED)[:3]
    JR = pin.getFrameJacobian(robot.model, robot.data, idR, pin.LOCAL_WORLD_ALIGNED)[:3]
    up = np.array([0., 0., load / 2.])
    return JL.T @ (force * u + up) + JR.T @ (-force * u + up)


def controllaw(sim, robot, trajs, tcurrent, cube):
    matchsimulatormodel(robot)
    q, vq = sim.getpybulletstate()
    q_of_t, vq_of_t, vvq_of_t = trajs[:3]
    qref, vqref, aqref = q_of_t(tcurrent), vq_of_t(tcurrent), vvq_of_t(tcurrent)

    aqdes = aqref + Kp * (qref - q) + Kv * (vqref - vq)
    M = pin.crba(robot.model, robot.data, q)
    h = pin.nle(robot.model, robot.data, q, vq)
    torques = M @ aqdes + h

    ramp = min(1., max(tcurrent, 0.) / GRASP_RAMP)
    force = squeezecommand(sim, ramp * GRASP_FORCE)
    torques += graspforcetorques(robot, q, force, ramp * cubemass(sim) * 9.81)
    sim.step(list(torques))

if __name__ == "__main__":

    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT

    robot, sim, cube = setupwithpybullet()


    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    from path import computepath

    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)


    #setting initial configuration
    sim.setqsim(q0)

    # display only: the table and obstacle meshes refer to .mtl files missing
    # from the repository, rendered black by some OpenGL drivers
    p = sim.p
    p.changeVisualShape(sim.tableId, -1, rgbaColor=[0.85, 0.75, 0.6, 1.])
    p.changeVisualShape(sim.obstacleId, -1, rgbaColor=[0.8, 0.1, 0.1, 1.])
    p.changeVisualShape(sim.cubeId, -1, rgbaColor=[0.95, 0.95, 0.95, 1.])
    targetshape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.001], rgbaColor=[0., 0.8, 0., 1.])
    targetpos = CUBE_PLACEMENT_TARGET.translation - np.array([0., 0., 0.048])
    p.createMultiBody(baseMass=0., baseVisualShapeIndex=targetshape, basePosition=list(targetpos))

    if not (successinit and successend) or len(path) < 3:
        print("error: no valid grasp configurations or path, the motion will not be valid")

    from trajectory import maketraj
    q_of_t, vq_of_t, vvq_of_t, total_time = maketraj(robot, cube, path)
    trajs = (q_of_t, vq_of_t, vvq_of_t)

    if RECORD_VIDEO:
        p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "demo.mp4")

    tcur = 0.

    # trajectory, then 1 s holding the final configuration
    while tcur < total_time + 1.:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        tcur += DT
    
    
    
