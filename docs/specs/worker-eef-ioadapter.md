# Worker end-effector force helpers

## Objective

Provide stateless MJX helpers that turn a world-frame end-effector pose delta
into generalized forces for the robot DOFs, including gripper DOFs.

## Assumptions

- The named end effector and robot base are MuJoCo bodies. The controlled point
  is the end-effector body frame origin; all pose, velocity, and wrench values
  are expressed in world coordinates.
- Position and quaternion deltas are already unnormalized and scaled. The
  quaternion is a unit relative world-frame rotation, not a componentwise
  quaternion difference.
- Desired end-effector velocity and acceleration are zero. `ke` and `kd` are
  proportional and derivative gains, each scalar or six-dimensional.
- Robot DOFs are those belonging to the base body and its descendants,
  including gripper joints. Environment and object DOFs are excluded. The
  Jacobian columns for gripper joints are taken from MJX without forcing them
  to zero.
- The force law uses equation (24) with the current bias force, passive force,
  and Jacobian derivative. Passive force is subtracted from bias because it
  acts on the same side of the dynamics equation as actuator force. Constraint
  forces are not compensated. There is no null-space force. The result is
  generalized force, not actuator control input.
- The MJX model and data describe the same, already-forwarded state. Body
  names are static per compiled model.

## API

`simulator/ioadapter.py` exposes the force conversion helper and the requested
Jacobian, inertia, bias, constraint-force, and task-space force helpers. Inputs
are MJX model/data and JAX arrays. Outputs are JAX arrays; the top-level result
has one entry per robot DOF in model order.
