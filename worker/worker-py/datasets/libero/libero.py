import os
from importlib.util import find_spec
from pathlib import Path
from xml.etree import ElementTree as ET


def _configure_libero_paths() -> None:
    """Create LIBERO's required path configuration without an input prompt."""
    config_directory = Path(
        os.environ.setdefault("LIBERO_CONFIG_PATH", "/tmp/libero")
    )
    config_file = config_directory / "config.yaml"

    vendored_benchmark_root = Path(
        os.environ.get(
            "LIBERO_BENCHMARK_ROOT",
            "/opt/lagrangian/third-party/libero/libero/libero",
        )
    )
    if vendored_benchmark_root.is_dir():
        benchmark_root = vendored_benchmark_root
    else:
        libero_spec = find_spec("libero.libero")
        if libero_spec is None or libero_spec.origin is None:
            raise ModuleNotFoundError("The LIBERO package is not installed")
        benchmark_root = Path(libero_spec.origin).parent

    dataset_directory = config_directory / "datasets"
    default_paths = {
        "benchmark_root": benchmark_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "datasets": dataset_directory,
        "assets": benchmark_root / "assets",
    }

    config_directory.mkdir(parents=True, exist_ok=True)
    dataset_directory.mkdir(exist_ok=True)
    config_contents = "".join(
        f"{name}: {path}\n" for name, path in default_paths.items()
    )
    if (
        not config_file.exists()
        or config_file.read_text(encoding="utf-8") != config_contents
    ):
        config_file.write_text(config_contents, encoding="utf-8")


_configure_libero_paths()

from libero.libero import benchmark
from libero.libero.envs.env_wrapper import ControlEnv


SUPPORTED_DATASETS = {
    "libero_10",
    "libero_90",
    "libero_goal",
    "libero_object",
    "libero_spatial",
}


def couple_panda_gripper_jaws(mjcf: str) -> str:
    """Couple Panda's opposing finger joints with one gripper actuator."""
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.fromstring(mjcf, parser=parser)
    joint_names = {
        joint.get("name") for joint in root.findall("./worldbody//joint")
    }
    finger_pairs = [
        (name, f"{name[:-1]}2")
        for name in joint_names
        if name is not None
        and name.endswith("finger_joint1")
        and f"{name[:-1]}2" in joint_names
    ]
    if len(finger_pairs) != 1:
        raise ValueError("Expected exactly one Panda gripper finger joint pair")
    first_joint, second_joint = finger_pairs[0]

    actuator_section = root.find("actuator")
    if actuator_section is None:
        raise ValueError("Panda gripper actuators are missing")
    gripper_actuators = [
        actuator
        for actuator in actuator_section
        if actuator.get("joint") in (first_joint, second_joint)
    ]
    target_joints = [actuator.get("joint") for actuator in gripper_actuators]
    if not 1 <= len(gripper_actuators) <= 2 or len(set(target_joints)) != len(
        target_joints
    ):
        raise ValueError("Expected one or two Panda gripper actuators on distinct joints")
    if len(gripper_actuators) == 2:
        # Keep the actuator on finger_joint1, whose target range is positive.
        actuator_section.remove(
            next(a for a in gripper_actuators if a.get("joint") == second_joint)
        )

    equality_section = root.find("equality")
    if equality_section is None:
        equality_section = ET.SubElement(root, "equality")
    existing = [
        constraint
        for constraint in equality_section.findall("joint")
        if {constraint.get("joint1"), constraint.get("joint2")}
        == {first_joint, second_joint}
    ]
    if existing:
        if len(existing) != 1 or list(
            map(float, existing[0].get("polycoef", "0 1 0 0 0").split())
        ) != [0, -1, 0, 0, 0]:
            raise ValueError("Panda gripper joints already have a conflicting equality")
    else:
        # The finger joint ranges have opposite signs, so q2 = -q1.
        ET.SubElement(
            equality_section,
            "joint",
            joint1=second_joint,
            joint2=first_joint,
            polycoef="0 -1 0 0 0",
        )
    return ET.tostring(root, encoding="unicode")


def get_task_mjcfs(
    task_indices: list[int],
    dataset: str,
    robot: str = None,
    robot_controller_config: dict = None,
) -> list[str]:
    """Return the complete MJCF model for each requested LIBERO task."""
    if dataset not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(
            f"Unsupported LIBERO dataset {dataset!r}; expected one of: {supported}"
        )

    task_suite = benchmark.get_benchmark_dict()[dataset]()
    task_count = task_suite.get_num_tasks()
    mjcfs = []
    robots = [robot]

    if robot is None:
        robots = ["Panda"]

    for task_index in task_indices:
        if isinstance(task_index, bool) or not isinstance(task_index, int):
            raise TypeError(f"LIBERO task indices must be integers; got {task_index!r}")
        if not 0 <= task_index < task_count:
            raise IndexError(
                f"LIBERO task index {task_index} is out of range for {dataset!r}; "
                f"expected 0 <= index < {task_count}"
            )

        environment = ControlEnv(
            bddl_file_name=task_suite.get_task_bddl_file_path(task_index),
            robots=robots,
            use_camera_obs=False,
            has_renderer=False,
            has_offscreen_renderer=False,
            controller_config=robot_controller_config,
        )
        try:
            mjcfs.append(environment.env.model.get_xml())
        finally:
            environment.close()

    return mjcfs
