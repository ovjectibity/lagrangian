/**
 * Local MLP testing: new weights, then a checkpoint.
 * Set ACCESS_KEY, MODEL_PATH and PROJECT_ID from a training run.
 * CHECKPOINT_ID selects a compatible policy checkpoint.
 * Node >= 22 and Python 3 are required; no SDK is used.
 * TEST_DRY_RUN=1 prints both current request fixtures.
 */
import {
  findMlpCheckpoint,
  jointInput,
  jointOutput,
  positiveInteger,
  prepareModel,
  runWorkload,
  zeroReward,
} from "./workload-api.mjs";

async function main() {
  const { projectId, modelId } = await prepareModel(
    "local-mlp-testing",
  );
  const timesteps = positiveInteger("NUM_TIMESTEPS", 50);
  const trajectories = positiveInteger("NUM_ENVS", 1);
  const input = jointInput();
  const model = {
    type: "MLP",
    content: {
      num_hidden_layers: 2,
      hidden_layer_dim: 64,
      input_dim: 55,
      output_dim: 16,
    },
  };
  for (const label of ["new-mlp", "checkpoint-mlp"]) {
    const checkpoint =
      label === "checkpoint-mlp"
        ? await findMlpCheckpoint(projectId)
        : null;
    const graph = {
      test_job_controller: {
        test_method: {
          type: "Custom",
          content: {
            test_params: {
              num_timesteps: timesteps,
              num_parallel_envs: trajectories,
            },
            simulation_model_id: modelId,
            reward_function: zeroReward(input),
            control_model: {
              model_type: checkpoint
                ? {
                    type: "Checkpoint",
                    content: {
                      type: "CheckpointID",
                      content: checkpoint.checkpoint_id,
                    },
                  }
                : { type: "New" },
              model: checkpoint?.model_spec ?? model,
              input_spec: input,
              output_spec: jointOutput(),
            },
          },
        },
        compute_instance: "AWSG4dn",
        acceleration_mode: "MLCPUSimulationCPU",
      },
    };
    await runWorkload("test", projectId, graph, label, {
      trajectories,
      timesteps,
    });
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
