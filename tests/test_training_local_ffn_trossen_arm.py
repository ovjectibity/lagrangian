"""Direct API training smoke test; requires ACCESS_KEY and MODEL_PATH.

Install tests/requirements.txt. PROJECT_ID reuses a project.
TEST_DRY_RUN=1 prints the current request without using services.
The reward is Python; local execution uses the Python CPU worker.
"""

from workload_api import positive_integer, prepare_model, run_workload

PYTHON_REWARD = r"""import jax.numpy as jnp

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
"""

INPUT_SPEC = {
    "type": "StandardInputSpec",
    "content": {
        "base_body_name": None,
        "states": [
            {
                "type": "InputStateControllable",
                "content": {
                    "rebase_body_tree": None,
                    "excluded_body_trees": [],
                    "state_item": {"type": kind},
                },
            }
            for kind in ("QPos", "QVel")
        ],
    },
}
OUTPUT_SPEC = {
    "type": "ProprioceptiveOutputSpec",
    "content": {
        "base_body_name": "base_link",
        "state": {
            "is_delta": False,
            "rebase_body": "base_link",
            "state_item": {"type": "QPos"},
        },
    },
}


def build_training_graph(model_id):
    reward_states = [
        {
            "type": "InputStateControllable",
            "content": {
                "rebase_body_tree": None,
                "excluded_body_trees": [],
                "state_item": {
                    "type": "XPose",
                    "content": {
                        "frame_spec": "WorldFrame",
                        "orientation_spec": "Quaternion",
                    },
                },
            },
        }
    ]
    reward_states.append({"type": "InputStateSensors", "content": "ContactPairs"})
    policy = {
        "type": "MLP",
        "content": {
            "num_hidden_layers": 2,
            "hidden_layer_dim": 64,
            "input_dim": 55,
            "output_dim": 16,
        },
    }
    value = {
        "type": "MLP",
        "content": {
            **policy["content"],
            "output_dim": 1,
        },
    }
    return {
        "training_job_controller": {
            "training_method": {
                "type": "PPO",
                "content": {
                    "num_trajs": positive_integer("NUM_TRAJS", 2),
                    "num_steps_per_traj": positive_integer("NUM_TIMESTEPS", 10),
                    "num_rollouts": positive_integer("NUM_ROLLOUTS", 1),
                    "num_epochs": positive_integer("NUM_EPOCHS", 2),
                    "epsilon": 0.2,
                    "gamma": 0.99,
                    "lambda": 0.95,
                    "reward_function": {
                        "language": "Python",
                        "reward_function_code": PYTHON_REWARD,
                        "input_spec": {
                            "type": "StandardInputSpec",
                            "content": {
                                "base_body_name": None,
                                "states": reward_states,
                            },
                        },
                    },
                    "control_model": {
                        "model_type": {"type": "New"},
                        "model": policy,
                        "input_spec": INPUT_SPEC,
                        "output_spec": OUTPUT_SPEC,
                    },
                    "value_model": {
                        "model_type": {"type": "New"},
                        "model": value,
                        "input_spec": INPUT_SPEC,
                    },
                    "simulation_model_id": model_id,
                    "control_output_mode": {"type": "ActionMean"},
                },
            },
            "acceleration_mode": "MLCPUSimulationCPU",
            "compute_instance": "AWSG4dn",
            "checkpoint_frequency": {"type": "LastOnly"},
        }
    }


def main():
    project, model = prepare_model("local-ffn-training")
    run_workload(project, build_training_graph(model), "local-ffn-training")


if __name__ == "__main__":
    main()
