/**
 * Complete LIBERO/Pi-0 trajectories through the current API.
 * Node >= 22 and ACCESS_KEY are required. PROJECT_ID is optional.
 * TASK_INDICES=0,1 NUM_ENVS=2 NUM_TIMESTEPS=20 sets coverage.
 * Every received camera PNG is saved in a timestamped run folder.
 * TEST_DRY_RUN=1 prints the request without using services.
 */
import {
  positiveInteger,
  prepareModel,
  runWorkload,
  vlaControl,
} from "./workload-api.mjs";

async function main() {
  const tasks = (process.env.TASK_INDICES ?? "0")
    .split(",")
    .map((value) => Number(value.trim()));
  if (
    !tasks.length ||
    new Set(tasks).size !== tasks.length ||
    tasks.some(
      (value) => !Number.isSafeInteger(value) || value < 0,
    ) ||
    (process.env.TASK_INDICES ?? "0")
      .split(",")
      .some((s) => !s.trim())
  ) {
    throw new Error(
      "TASK_INDICES must contain unique nonnegative integers",
    );
  }
  const environments = positiveInteger("NUM_ENVS", 1);
  const timesteps = positiveInteger("NUM_TIMESTEPS", 20);
  const { projectId } = await prepareModel(
    "libero-pi0-testing",
    false,
  );
  const graph = {
    test_job_controller: {
      test_method: {
        type: "Dataset",
        content: {
          dataset: {
            type: "Libero",
            content: {
              tasks_index: tasks,
              libero_dataset_type:
                process.env.LIBERO_DATASET ?? "LiberoSpatial",
              num_envs_per_task: environments,
              num_timesteps: timesteps,
            },
          },
          control_model: vlaControl(
            process.env.ROBOT_BASE_BODY ?? "robot0_base",
            process.env.EEF_BODY ?? "gripper0_right_eef",
            // Drive one finger; the equality moves the other.
            process.env.GRIPPER_BODY ??
              "gripper0_right_leftfinger",
          ),
        },
      },
      compute_instance: "AWSG4dn",
      acceleration_mode: "MLGPUSimulationGPU",
    },
  };
  await runWorkload(
    "test",
    projectId,
    graph,
    "libero-pi0-testing",
    {
      tasks,
      trajectories: environments,
      timesteps,
      cameras: true,
    },
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
