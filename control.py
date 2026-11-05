#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau

Part 2.II - Torque control law.

Inverse dynamics (computed torque) control to track the reference trajectory
q_ref(t), plus a grasping force applied by both hands on the cube, as
recommended in the instructions:

    ddq_des = ddq_ref + Kp (q_ref - q) + Kv (vq_ref - vq)          (PD + feedforward)
    tau     = M(q) ddq_des + h(q, vq)                               (inverse dynamics)
            + J_L(q)^T f_L + J_R(q)^T f_R                           (grasping force)

M is the joint space mass matrix (pin.crba) and h the nonlinear effects
(Coriolis, centrifugal and gravity: pin.nle). J_L, J_R are the linear parts
of the hand jacobians expressed in the world frame, and f_L = -f_R is a force
of magnitude GRASP_FORCE pushing each hand towards the other one, i.e.
squeezing the cube. Because the inverse dynamics model does not know about
the cube, this force is not compensated: it presses the hands on the cube
faces and naturally absorbs the small alignment errors between the hands and
the cube, and the friction then carries the cube (mass 0.14 kg, friction 1.0,
so any force above ~0.7 N per hand would hold it; I use a margin).
"""

import numpy as np
import pinocchio as pin

from bezier import Bezier
from config import LEFT_HAND, RIGHT_HAND

# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 300.               # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)   # derivative gain (D of PD)

GRASP_FORCE  = 30.      # (N) squeezing force applied by each hand on the cube
GRASP_RAMP   = 0.3      # (s) the grasping force is ramped up over this duration


def matchsimulatormodel(robot):
    '''Make the pinocchio dynamic model consistent with pybullet.
    The effector links LARM_EFF_Link / RARM_EFF_Link have no <inertial> tag
    in the urdf. Pinocchio then gives them a zero mass, but pybullet gives
    them a mass of 1 kg (see the pybullet warning "No inertial data for link,
    using mass=1" when the robot is loaded). Checked with
    pybullet.getDynamicsInfo: mass 1.0, local inertia diagonal (0, 0, 0), i.e.
    a 1 kg point mass at the effector frame.
    Without this correction the inverse dynamics model misses 1 kg at each
    wrist and the arms sag (steady tracking error ~0.2 rad).
    Done once per robot (the flag avoids adding the mass twice).'''
    if getattr(robot, '_simmodelmatched', False):
        return
    m = robot.model
    for name in (LEFT_HAND, RIGHT_HAND):
        frame = m.frames[m.getFrameId(name)]
        Ilink = pin.Inertia(1., np.zeros(3), np.zeros((3, 3)))  # point mass, as in pybullet
        m.inertias[frame.parent] += frame.placement.act(Ilink)
    robot.data = m.createData()
    robot._simmodelmatched = True


def cubemass(sim):
    '''Total mass of the cube in pybullet (base + hook links). The hook links
    of cube_small.urdf have no <inertial> tag either, so pybullet gives them
    1 kg each: the simulated cube weighs 2.14 kg, not 0.14 kg.'''
    if not hasattr(sim, '_cubemass'):
        p = sim.p
        sim._cubemass = p.getDynamicsInfo(sim.cubeId, -1)[0] + sum(
            p.getDynamicsInfo(sim.cubeId, i)[0] for i in range(p.getNumJoints(sim.cubeId)))
    return sim._cubemass


def graspforcetorques(robot, q, force, load=0.):
    '''Joint torques J_L^T f_L + J_R^T f_R where
      - f_L = -f_R has magnitude force and points from the left to the right
        hand: the hands squeeze the cube
      - plus, on each hand, an upward force load/2: feedforward compensation of
        the weight of the cube (load = m_cube * g), which the inverse dynamics
        model of the robot does not include.'''
    pin.computeJointJacobians(robot.model, robot.data, q)
    pin.framesForwardKinematics(robot.model, robot.data, q)
    idL = robot.model.getFrameId(LEFT_HAND)
    idR = robot.model.getFrameId(RIGHT_HAND)
    pL = robot.data.oMf[idL].translation
    pR = robot.data.oMf[idR].translation
    u = pR - pL
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

    # desired acceleration: feedforward + PD on the tracking error
    aqdes = aqref + Kp * (qref - q) + Kv * (vqref - vq)

    # inverse dynamics: tau = M(q) aq_des + h(q, vq)
    M = pin.crba(robot.model, robot.data, q)
    h = pin.nle(robot.model, robot.data, q, vq)
    torques = M @ aqdes + h

    # grasping force and cube weight compensation, both ramped up at the start
    # to avoid a shock on the cube
    ramp = min(1., max(tcurrent, 0.) / GRASP_RAMP)
    torques += graspforcetorques(robot, q, ramp * GRASP_FORCE, ramp * cubemass(sim) * 9.81)

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


    # Trajectory: level 1 (QP fit of a Bezier curve) with an automatic fallback
    # to level 0 (quintic time scaling along the path), see trajectory.py.
    # The simple example below (straight Bezier from q0 to qe, which does not
    # follow the path) is kept for reference only.
    def maketraj(q0,q1,T):
        q_of_t = Bezier([q0,q0,q1,q1],t_max=T)
        vq_of_t = q_of_t.derivative(1)
        vvq_of_t = vq_of_t.derivative(1)
        return q_of_t, vq_of_t, vvq_of_t

    from trajectory import maketraj as maketrajfrompath
    q_of_t, vq_of_t, vvq_of_t, total_time = maketrajfrompath(robot, cube, path)
    trajs = (q_of_t, vq_of_t, vvq_of_t)

    tcur = 0.

    # run the trajectory, then hold the final configuration for 1 s so that
    # the cube settles at the target
    while tcur < total_time + 1.:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        tcur += DT



