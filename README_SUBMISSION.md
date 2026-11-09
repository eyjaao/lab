# ARO lab

## Software
Only the course environment (`environment.yml`); quadprog and scipy are in `requirements.txt`.

## Running
- `python inverse_geometry.py`: q0 and qe
- `python path.py`: path displayed in meshcat
- `python control.py`: motion in pybullet (`RECORD_VIDEO = True` saves `demo.mp4`)
- `python test_robustness.py N`: N noisy runs

## Algorithms
**Inverse geometry:** damped least-squares IK on the 6D error (SE(3) log) of both hands, a postural task in the null space, and projection onto the joint limits. Colliding results are retried from perturbed, then random, seeds. Grasps whose hand-fixed wrist link hits the obstacle are rejected immediately.

**Motion planning:** RRT-Connect over grasping configurations. Cube placements sampled near the start and goal are solved with inverse geometry. Edges interpolate q and the cube, projecting each step back onto the grasp. A 2 cm obstacle clearance applies away from the endpoints; then random shortcutting.

**Trajectory:** level 1 is a quadprog least-squares Bezier fit with zero end velocities and accelerations, control points within the joint limits (convex hull), and duration from the velocity limits. Level 0 (quintic time scaling) is used if the fit collides.

**Control:** inverse dynamics (`pin.crba`, `pin.nle`) with PD and feedforward, plus a squeezing force through the hand Jacobians. The model is corrected for the 1 kg masses pybullet gives links without inertial data (effectors, cube hooks).

## Extensions
- Trajectory optimisation, level 1
- Robustness: `test_robustness.py`. In 20 noisy runs, all 9 that found a path succeeded in pybullet. Failures were goals pushed towards the obstacle (the default goal has 2 mm clearance).
- k-d tree (scipy `cKDTree`) for RRT nearest neighbours
- Grasp force feedback: integral control on the contact normal forces measured in pybullet. It held 30 N within 0.7 N RMS, similar to open-loop.
- Not implemented: level 2, Part 3
