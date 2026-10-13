import os
from importlib.util import find_spec
from pathlib import Path


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
