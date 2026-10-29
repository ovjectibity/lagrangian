# Worker end-effector force helpers implementation plan

1. Resolve body IDs and robot DOF indices from static MJX model metadata.
2. Use `mjx.jac` and the pinned MJX Jacobian-derivative helper for the named
   end-effector body origin; select robot columns and keep linear rows first.
3. Use `mjx.full_m` for the model inertia and select the robot block. Use JAX
   NumPy for the inverse, task-space inertia, bias, passive, and force
   calculations. Subtract `qfrc_passive` from `qfrc_bias` in the compensation
   term.
4. Convert world-frame pose deltas to six-dimensional pose error, apply PD
   gains with zero desired velocity and acceleration, then apply equation (24).
5. Map the resulting task-space wrench through the Jacobian transpose to robot
   generalized forces. Do not map those forces to actuator controls yet.
6. Check syntax and static structure locally; run physics tests when an
   environment with MuJoCo and JAX is available.
