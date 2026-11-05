# ARO lab: implementation notes

Draft implementation of Part 1 (geometry) and Part 2 (dynamics). Every file
keeps the original function names and signatures (`computeqgrasppose`,
`computepath`, `controllaw`).

## Results (default scene, tested headless)

| Step | Result |
|---|---|
| `computeqgrasppose` q0 / qe | both found, collision free, within joint limits, hand–hook error < 1 mm |
| IG on random cube targets | 29 / 36 solved (failures: cube tucked against the chest or the obstacle wall) |
| `computepath` | ~10 s, 7 RRT nodes, 216 configurations, all collision free and grasping (error < 0.5 mm), max step 0.023 rad |
| Path clearance from obstacle | ≥ 2 cm except within 15 cm of the start / goal |
| Trajectory level 1 (QP Bezier) | T ≈ 11 s, 0 collisions, 0 joint-limit violations |
| Trajectory level 0 (quintic) | T ≈ 10 s, 0 collisions |
| Control in pybullet | cube lifted to ~1.15 m and placed 5 mm from the target; max tracking error 0.04 rad |

## Files

### `inverse_geometry.py` (Part 1.I)
- `handserror`: 12D error (SE(3) log of handMhook for both hands, as in the
  6D IG of lab 0) and the stacked LOCAL frame Jacobians.
- `_solveik`: damped least squares (Gauss–Newton with Levenberg–Marquardt damping),
  plus a postural task projected in the null space of the hands task, plus
  projection onto the joint limits each step.
- `_posturalreference`: hand-picked "arms forward, elbows bent" posture (the
  hint about postural bias). **Tune this if the poses look unnatural in meshcat.**
- `computeqgrasppose`: tries qcurrent, then the posture, then 40 restarts around
  the posture (bias = the seed itself), then 200 uniformly random restarts.
  Returns the first valid solution (collision free + joint limits).
- `computeqgrasppose_fast`: single attempt from qcurrent, used inside the planner.
- `obstacleclearance`, `isvalid(robot, q, margin)`: collision / limits / clearance checks.

### `solution.py`
- `LMRREF`: placement of the right hand in the left hand frame when grasping
  = (0, −0.1, 0) with a rotation of −3.14 rad about z (from the hook frames
  in `cube_small.urdf`). Its `__main__` checks it against q0 (difference 4e-4).

### `path.py` (Part 1.II)
- **II.a** `samplecubeplacement` / `sampleconfiguration`: cube position uniform in a
  box spanning start and goal (±15 cm, up to 35 cm higher), orientation on
  the geodesic between start and goal orientations, then IG.
- **II.b** `projectpath`: interpolates q *and* the cube placement, and projects
  each interpolated q onto the grasp constraint (IG seeded with the
  interpolated q). Fails on IG failure, collision, or a jump > 0.15 rad
  (IK branch change). Returns (valid part of path, flag).
- **II.c** `rrtconnect`: RRT-Connect (lecture / Kuffner & LaValle) over grasping
  configurations; extensions steer the cube ≤ 10 cm; edges validated with
  `projectpath`. Then `shortcut` (30 random shortcuts) and `densify`.
- `requiredclearance`: 2 cm clearance from the obstacle along the path,
  ramped to 0 within 15 cm of start/goal (the goal pose itself is only ~2 mm
  from the obstacle).
- `computepath` has no robot/cube argument, so the module loads its own
  models once (`getmodels`).

### `trajectory.py` (new, Part 2.I)
- Level 0 `maketraj_level0`: quintic time scaling s(t) along the arc-length
  parametrised path (zero velocity and acceleration at both ends).
- Level 1 `maketraj_level1` / `fitbezier`: QP (quadprog) fit of a degree-49
  Bezier curve; equality constraints = 3 first/last control points (zero
  vel/acc, exact start/end); inequalities = control points within joint limits
  (convex hull → whole curve within limits); small smoothness regulariser
  (2nd differences). Duration T from the joint velocity limits (50 %).
- `maketraj`: level 1 if the trajectory doesn't hit the table/obstacle (checked
  with pinocchio), increasing control points if needed, else level 0.
- Note: T is computed from the sampled peak velocity; the convex-hull bound
  (`TIGHT_T = False`) is guaranteed but ~2× too conservative.

### `control.py` (Part 2.II)
- Inverse dynamics: `ddq_des = ddq_ref + Kp e + Kv de`,
  `tau = M ddq_des + h` (`pin.crba`, `pin.nle`), Kp = 300, Kv = 2√Kp.
- Grasp: `J_L^T f_L + J_R^T f_R` with a 30 N squeeze, plus a feedforward
  upward force m_cube·g/2 per hand; ramped over 0.3 s.
- Two model fixes found by testing:
  1. pybullet gives the effector links (no `<inertial>` in the URDF) 1 kg
     point masses; pinocchio gives 0. `matchsimulatormodel` adds them.
     Without it: ~0.2 rad sag at rest.
  2. The cube's hook links likewise get 1 kg each → the simulated cube is
     2.14 kg (`cubemass`). Without compensation: ~0.09 rad error while
     carrying, and the left wrist caught the obstacle.
- `__main__` runs the trajectory then holds for 1 s.

## Things to check / refine
- Look at q0 / qe and the path in meshcat (`python path.py`): tune
  `_posturalreference` if the arms look odd (elbows are wider than in
  `images/q0.png`).
- Parameters to try: `CLEARANCE_RAMP`, `PATH_CLEARANCE`, `GRASP_FORCE`, `N_CTRL`.
- Run times: IG ≤ 2 s on hard targets, path ~10 s.
- Optional / not done: level 2 (collision-aware trajectory optimisation), Part 3.
- Cite sources in code as required (lecture, tutorials, Kuffner & LaValle, and
  how you used AI assistance, per course policy).
