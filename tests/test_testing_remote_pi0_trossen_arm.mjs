/**
 * Pi-0 custom-model smoke test using direct HTTP and WebSocket.
 * Node >= 22, Python 3, ACCESS_KEY and MODEL_PATH are required.
 * NUM_ENVS and NUM_TIMESTEPS configure full trajectory coverage.
 * TEST_DRY_RUN=1 prints the request without using services.
 */
import {
  jointInput,
  positiveInteger,
  prepareModel,
  runWorkload,
  vlaControl,
  zeroReward,
} from "./workload-api.mjs";

async function main() {
  const environments = positiveInteger("NUM_ENVS", 1);
  const timesteps = positiveInteger("NUM_TIMESTEPS", 30);
  const base = process.env.ROBOT_BASE_BODY ?? "base_link";
  const eef = process.env.EEF_BODY ?? "gripper_link";
  const gripper = process.env.GRIPPER_BODY ?? "left_finger_link";
  const { projectId, modelId } = await prepareModel(
    "remote-pi0-testing",
  );
  const graph = {
    test_job_controller: {
      test_method: {
        type: "Custom",
        content: {
          test_params: {
            num_timesteps: timesteps,
            num_parallel_envs: environments,
          },
          simulation_model_id: modelId,
          control_model: vlaControl(base, eef, gripper),
          reward_function: zeroReward(jointInput(base, ["QPos"])),
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
    "remote-pi0-testing",
    {
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
