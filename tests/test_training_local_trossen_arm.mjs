/**
 * Local (AllOnServer) training test with a Rust reward function.
 * Mirrors tests/test_training_local_trossen_arm.py using the TypeScript SDK.
 *
 * Usage:
 *   MODEL_PATH=assets/trossen_vx300s/vx300s.xml node tests/test_training_local_trossen_arm.mjs
 *   BASE_URL=http://localhost:8000/api MODEL_PATH=... node tests/test_training_local_trossen_arm.mjs
 *
 * Requires: node >= 18, sdk built (cd sdk && npm run build)
 */

import { execSync } from "node:child_process";
import { readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { LagrangianClient } from "../sdk/dist/index.js";

const BASE_URL = process.env.BASE_URL ?? "http://localhost:8000/api";
const MODEL_PATH = process.env.MODEL_PATH ?? "";

if (!MODEL_PATH) {
  console.error("MODEL_PATH env var is required");
  process.exit(1);
}

const RUST_REWARD = `
pub fn reward(
    prev_cstate: &[f64],
    curr_cstate: &[f64],
    action: &[f64],
    contact_pairs: &[(usize, usize)],
) -> f64 {
    let target_cube: usize = 3;
    let gripper: usize = 10;
    let left_finger: usize = 12;
    let right_finger: usize = 13;
    let left_box: usize = 1;
    let right_box: usize = 2;

    let get_pos = |s: &[f64], i: usize| -> (f64, f64, f64) {
        (s[i * 3], s[i * 3 + 1], s[i * 3 + 2])
    };

    let target_xy = (0.35_f64, -0.22_f64);
    let left_box_start = (0.35_f64, 0.22_f64);
    let right_box_start = (0.35_f64, -0.22_f64);

    let (nx, ny, nz) = get_pos(curr_cstate, target_cube);
    let (px, py, _) = get_pos(prev_cstate, target_cube);
    let (gx, gy, _) = get_pos(curr_cstate, gripper);

    let next_dist = ((nx - target_xy.0).powi(2) + (ny - target_xy.1).powi(2)).sqrt();
    let prev_dist = ((px - target_xy.0).powi(2) + (py - target_xy.1).powi(2)).sqrt();
    let r_prog = (prev_dist - next_dist) * 10.0;

    let gripper_xy_dist = ((gx - nx).powi(2) + (gy - ny).powi(2)).sqrt();
    let r_gripper = -2.0 * gripper_xy_dist;

    let (lx, ly, _) = get_pos(curr_cstate, left_box);
    let (rx, ry, _) = get_pos(curr_cstate, right_box);
    let left_disp = ((lx - left_box_start.0).powi(2) + (ly - left_box_start.1).powi(2)).sqrt();
    let right_disp = ((rx - right_box_start.0).powi(2) + (ry - right_box_start.1).powi(2)).sqrt();
    let r_box = -20.0 * (left_disp + right_disp);

    let cube_drop = if nz < 0.10 { -20.0 } else { 0.0 };

    let grip_contact: f64 = contact_pairs.iter().filter(|(a, b)| {
        (*a == target_cube && (*b == left_finger || *b == right_finger))
            || (*b == target_cube && (*a == left_finger || *a == right_finger))
    }).count() as f64 * 20.0;

    r_prog + r_gripper + r_box + cube_drop + grip_contact
}
`;

function buildTrainingGraph(modelId) {
  return {
    training_job_controller: {
      training_method: {
        type: "PPO",
        content: {
          num_trajs: 2,
          num_steps_per_traj: 10,
          num_epochs: 1,
          num_updates: 1,
          epsilon: 0.2,
          gamma: 0.99,
          lambda: 0.95,
          reward_function: {
            language: "Rust",
            reward_function_code: RUST_REWARD,
          },
          control_model: {
            type: "MLP",
            content: {
              num_hidden_layers: 2,
              hidden_layer_dim: 64,
              output_dim: 14,
              input_dim: 55,
            },
          },
          value_model: {
            type: "MLP",
            content: {
              num_hidden_layers: 2,
              hidden_layer_dim: 64,
              output_dim: 1,
              input_dim: 55,
            },
          },
          simulation_model: modelId,
          control_output_mode: { type: "ActionMean" },
        },
      },
      acceleration_mode: "AllOnServer",
      compute_instance: "AWSG4dn",
      checkpoint_frequency: { type: "LastOnly" },
    },
  };
}

async function uploadModel(client) {
  const modelDir = dirname(resolve(MODEL_PATH));
  const zipPath = join(tmpdir(), `lagrangian-model-${Date.now()}.zip`);
  try {
    execSync(`zip -r "${zipPath}" .`, { cwd: modelDir, stdio: "pipe" });
    const buf = readFileSync(zipPath);
    const blob = new Blob([buf], { type: "application/zip" });
    const result = await client.models.upload(blob, "vx300s.zip");
    console.log(`  model_id=${result.id}`);
    return result.id;
  } finally {
    rmSync(zipPath, { force: true });
  }
}

async function main() {
  const client = new LagrangianClient({ baseUrl: BASE_URL });

  console.log("=== Uploading model ===");
  const modelId = await uploadModel(client);

  console.log("=== Creating experiment ===");
  const experiment = await client.experiments.create({ name: "reward-fn-test" });
  const expId = experiment.training_experiment_id;
  console.log(`  exp_id=${expId}`);

  console.log("\n=== Local training with Rust reward ===");
  const run = await client.experiments.createRun(expId);
  const runId = run.id;
  console.log(`  run_id=${runId}`);

  const graph = buildTrainingGraph(modelId);
  await client.experiments.startTraining(expId, runId, graph);
  console.log("  start_training: ok");

  console.log("  Streaming SSE events:");
  let count = 0;
  for await (const event of client.experiments.streamTrainingUpdates(expId, runId)) {
    const preview = JSON.stringify(event).slice(0, 120);
    console.log(`    SSE[${count}]: ${preview}`);
    count++;
    if (count >= 3) break;
  }

  console.log("\n=== All tests passed ===");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
