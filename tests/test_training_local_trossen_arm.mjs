/**
 * Local Python-worker training smoke test using direct HTTP/WS.
 * Node >= 22, Python 3, ACCESS_KEY and MODEL_PATH are required.
 * PROJECT_ID reuses a project. TEST_DRY_RUN=1 prints a fixture.
 * Python counterparts use the same contracts.
 */
import {
  jointInput,
  jointOutput,
  positiveInteger,
  prepareModel,
  runWorkload,
} from "./workload-api.mjs";

const PYTHON_REWARD = `import jax.numpy as jnp

def reward(state_pre, state_post):
    previous = state_pre.content.states[0].content.state_item
    current = state_post.content.states[0].content.state_item
    pre = previous.reshape(-1, 7)[:, :3]
    post = current.reshape(-1, 7)[:, :3]
    target = jnp.array([0.35, -0.22])
    progress = 10.0 * (
        jnp.linalg.norm(pre[3, :2] - target)
        - jnp.linalg.norm(post[3, :2] - target)
    )
    reach = -2.0 * jnp.linalg.norm(post[10, :2] - post[3, :2])
    boxes = -20.0 * (
        jnp.linalg.norm(post[1, :2] - jnp.array([0.35, 0.22]))
        + jnp.linalg.norm(post[2, :2] - target)
    )
    drop = jnp.where(post[3, 2] < 0.10, -20.0, 0.0)
    pairs = state_post.content.states[1].content.reshape(-1, 2)
    finger_a = (pairs[:, 0] == 12) | (pairs[:, 0] == 13)
    finger_b = (pairs[:, 1] == 12) | (pairs[:, 1] == 13)
    touching = ((pairs[:, 0] == 3) & finger_b) | (
        (pairs[:, 1] == 3) & finger_a
    )
    grip = 20.0 * jnp.sum(touching)
    return progress + reach + boxes + drop + grip
`;

function buildTrainingGraph(modelId) {
  const input = jointInput();
  const policy = {
    type: "MLP",
    content: {
      num_hidden_layers: 2,
      hidden_layer_dim: 64,
      input_dim: 55,
      output_dim: 16,
    },
  };
  const rewardStates = [
    {
      type: "InputStateControllable",
      content: {
        rebase_body_tree: null,
        excluded_body_trees: [],
        state_item: {
          type: "XPose",
          content: {
            frame_spec: "WorldFrame",
            orientation_spec: "Quaternion",
          },
        },
      },
    },
    { type: "InputStateSensors", content: "ContactPairs" },
  ];
  return {
    training_job_controller: {
      training_method: {
        type: "PPO",
        content: {
          num_trajs: positiveInteger("NUM_TRAJS", 2),
          num_steps_per_traj: positiveInteger(
            "NUM_TIMESTEPS",
            10,
          ),
          num_rollouts: positiveInteger("NUM_ROLLOUTS", 1),
          num_epochs: positiveInteger("NUM_EPOCHS", 2),
          epsilon: 0.2,
          gamma: 0.99,
          lambda: 0.95,
          reward_function: {
            language: "Python",
            reward_function_code: PYTHON_REWARD,
            input_spec: {
              type: "StandardInputSpec",
              content: {
                base_body_name: null,
                states: rewardStates,
              },
            },
          },
          control_model: {
            model_type: { type: "New" },
            model: policy,
            input_spec: input,
            output_spec: jointOutput(),
          },
          value_model: {
            model_type: { type: "New" },
            model: {
              type: "MLP",
              content: {
                ...policy.content,
                output_dim: 1,
              },
            },
            input_spec: input,
          },
          simulation_model_id: modelId,
          control_output_mode: { type: "ActionMean" },
        },
      },
      acceleration_mode: "MLCPUSimulationCPU",
      compute_instance: "AWSG4dn",
      checkpoint_frequency: { type: "LastOnly" },
    },
  };
}

async function main() {
  const { projectId, modelId } =
    await prepareModel("local-training");
  const graph = buildTrainingGraph(modelId);
  const params =
    graph.training_job_controller.training_method.content;
  await runWorkload(
    "training",
    projectId,
    graph,
    "local-training",
    {
      rollouts: params.num_rollouts,
      trajectories: params.num_trajs,
      timesteps: params.num_steps_per_traj,
    },
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
