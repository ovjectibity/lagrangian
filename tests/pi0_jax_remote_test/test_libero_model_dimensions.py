#!/usr/bin/env python3
"""Build a Panda LIBERO task with joint-position control and inspect its dimensions.

Run with run_remote_test.sh. LIBERO_DATASET and LIBERO_TASK_INDEX select the
task (libero_spatial, 0 by default).
"""

import os

import mujoco
import robosuite

from datasets.libero.libero import get_task_mjcfs


def main() -> None:
    dataset = os.environ.get("LIBERO_DATASET", "libero_spatial")
    task_index = int(os.environ.get("LIBERO_TASK_INDEX", "0"))

    controller_config = robosuite.load_composite_controller_config(robot="Panda")
    arm_config = robosuite.load_part_controller_config(
        default_controller="JOINT_POSITION"
    )
    arm_config["gripper"] = controller_config["body_parts"]["right"]["gripper"]
    controller_config["body_parts"]["right"] = arm_config

    (mjcf,) = get_task_mjcfs(
        [task_index],
        dataset,
        robot="Panda",
        robot_controller_config=controller_config,
    )
    model = mujoco.MjModel.from_xml_string(mjcf)

    print(
        f"task={dataset}[{task_index}] robot=Panda "
        f"controller=JOINT_POSITION nu={model.nu} nv={model.nv} nq={model.nq}",
        flush=True,
    )
    children = [[] for _ in range(model.nbody)]
    for body_id in range(1, model.nbody):
        parent_id = int(model.body_parentid[body_id])
        children[parent_id].append(body_id)

    print("Body hierarchy:", flush=True)
    stack = [(0, "", True)]
    while stack:
        body_id, prefix, is_last = stack.pop()
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        branch = "" if body_id == 0 else ("`-- " if is_last else "|-- ")
        print(f"{prefix}{branch}body[{body_id}]={body_name}", flush=True)

        child_prefix = prefix + ("    " if is_last else "|   ") if body_id else ""
        joint_start = int(model.body_jntadr[body_id])
        joint_count = int(model.body_jntnum[body_id])
        for joint_offset in range(joint_count):
            joint_id = joint_start + joint_offset
            joint_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            joint_type = mujoco.mjtJoint(int(model.jnt_type[joint_id]))
            parent_id = int(model.body_parentid[body_id])
            parent_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, parent_id
            )
            joint_is_last = joint_offset == joint_count - 1 and not children[body_id]
            joint_branch = "`-- " if joint_is_last else "|-- "
            print(
                f"{child_prefix}{joint_branch}joint[{joint_id}]={joint_name} "
                f"type={joint_type.name.removeprefix('mjJNT_')} "
                f"connects=body[{parent_id}]={parent_name}->body[{body_id}]={body_name}",
                flush=True,
            )

        for reversed_index, child_id in enumerate(reversed(children[body_id])):
            stack.append((child_id, child_prefix, reversed_index == 0))

    print("Actuators (geoms are directly attached to the target's body):", flush=True)
    joint_transmissions = {
        mujoco.mjtTrn.mjTRN_JOINT,
        mujoco.mjtTrn.mjTRN_JOINTINPARENT,
    }
    for actuator_id in range(model.nu):
        actuator_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
        )
        transmission = mujoco.mjtTrn(int(model.actuator_trntype[actuator_id]))
        target_id = int(model.actuator_trnid[actuator_id, 0])
        if transmission in joint_transmissions:
            joint_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, target_id
            )
            target = f"joint[{target_id}]={joint_name}"
            body_id = int(model.jnt_bodyid[target_id])
        elif transmission == mujoco.mjtTrn.mjTRN_SITE:
            site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, target_id)
            target = f"site[{target_id}]={site_name}"
            body_id = int(model.site_bodyid[target_id])
            refsite_id = int(model.actuator_trnid[actuator_id, 1])
            if refsite_id >= 0:
                refsite_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_SITE, refsite_id
                )
                ref_body_id = int(model.site_bodyid[refsite_id])
                ref_body_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, ref_body_id
                )
                target += (
                    f" refsite[{refsite_id}]={refsite_name} "
                    f"ref_body[{ref_body_id}]={ref_body_name}"
                )
        else:
            print(
                f"actuator[{actuator_id}]={actuator_name} "
                f"transmission={transmission.name} target_id={target_id} "
                "target_details=n/a",
                flush=True,
            )
            continue

        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        geom_start = int(model.body_geomadr[body_id])
        geom_end = geom_start + int(model.body_geomnum[body_id])
        body_geoms = [
            f"geom[{geom_id}]={mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)}"
            for geom_id in range(geom_start, geom_end)
        ]
        print(
            f"actuator[{actuator_id}]={actuator_name} "
            f"transmission={transmission.name} "
            f"{target} "
            f"body[{body_id}]={body_name} body_geoms={body_geoms}",
            flush=True,
        )

    print(f"Equality constraints ({model.neq}):", flush=True)
    for equality_id in range(model.neq):
        equality_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_id
        )
        equality_type = mujoco.mjtEq(int(model.eq_type[equality_id]))
        object_type = mujoco.mjtObj(int(model.eq_objtype[equality_id]))
        object1_id = int(model.eq_obj1id[equality_id])
        object2_id = int(model.eq_obj2id[equality_id])
        object1_name = mujoco.mj_id2name(model, object_type, object1_id)
        object2_name = (
            mujoco.mj_id2name(model, object_type, object2_id)
            if object2_id >= 0
            else None
        )
        print(
            f"equality[{equality_id}]={equality_name} "
            f"type={equality_type.name} object_type={object_type.name} "
            f"object1[{object1_id}]={object1_name} "
            f"object2[{object2_id}]={object2_name} "
            f"active0={bool(model.eq_active0[equality_id])} "
            f"data={model.eq_data[equality_id].tolist()}",
            flush=True,
        )

    if model.nu != 9:
        raise AssertionError(f"Expected 9 Panda actuators, got nu={model.nu}")
    if not model.nq >= model.nv > 0:
        raise AssertionError(
            f"Invalid model dimensions: nq={model.nq}, nv={model.nv}"
        )


if __name__ == "__main__":
    main()
