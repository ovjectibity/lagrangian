from libero.libero import benchmark
from libero.libero.envs.env_wrapper import ControlEnv


SUPPORTED_DATASETS = {
    "libero_10",
    "libero_90",
    "libero_goal",
    "libero_object",
    "libero_spatial",
}


def get_task_mjcfs(task_indices: list[int], dataset: str) -> list[str]:
    """Return the complete MJCF model for each requested LIBERO task."""
    if dataset not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(
            f"Unsupported LIBERO dataset {dataset!r}; expected one of: {supported}"
        )

    task_suite = benchmark.get_benchmark_dict()[dataset]()
    task_count = task_suite.get_num_tasks()
    mjcfs = []

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
            robots=["Panda"],
            use_camera_obs=False,
            has_renderer=False,
            has_offscreen_renderer=False,
        )
        try:
            mjcfs.append(environment.env.model.get_xml())
        finally:
            environment.close()

    return mjcfs
