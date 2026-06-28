/**
 * Local (AllOnServer) testing workload tests using the TypeScript SDK.
 *
 * Runs two back-to-back test cases:
 *   1. Local testing | Vanilla MLP (random weights)
 *   2. Local testing | MLP from checkpoint
 *
 * Usage:
 *   MODEL_PATH=assets/trossen_vx300s/vx300s.xml node tests/test_testing_local_mlp_trossen_arm.mjs
 *   BASE_URL=http://localhost:8000/api MODEL_PATH=... CHECKPOINT_ID=<id> node tests/test_testing_local_mlp_trossen_arm.mjs
 *
 * Requires: node >= 18, sdk built (cd sdk && npm run build)
 * Set CHECKPOINT_ID to use a specific checkpoint; otherwise the latest MLP checkpoint is fetched.
 */

import { spawnSync } from "node:child_process";
import { readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { LagrangianClient } from "../sdk/dist/index.js";

const BASE_URL      = process.env.BASE_URL      ?? "http://localhost:8000/api";
const MODEL_PATH    = process.env.MODEL_PATH     ?? "";
const CHECKPOINT_ID = process.env.CHECKPOINT_ID  ?? "";

const MLP_INPUT_DIM  = 55;
const MLP_OUTPUT_DIM = 14;
const NUM_TIMESTEPS  = 50;
const MAX_FRAMES     = 5;

if (!MODEL_PATH) {
  console.error("MODEL_PATH env var is required");
  process.exit(1);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

async function uploadModel(client) {
  const modelDir = dirname(resolve(MODEL_PATH));
  const zipPath = join(tmpdir(), `lagrangian-model-${Date.now()}.zip`);
  try {
    const r = spawnSync("zip", ["-r", zipPath, "."], { cwd: modelDir });
    if (r.error) throw r.error;
    if (r.status !== 0) throw new Error(`zip failed (${r.status}): ${r.stderr?.toString()}`);
    const blob = new Blob([readFileSync(zipPath)], { type: "application/zip" });
    const result = await client.models.upload(blob, "vx300s.zip");
    console.log(`  model_id=${result.id}`);
    return result.id;
  } finally {
    rmSync(zipPath, { force: true });
  }
}

async function fetchMlpCheckpoint(client) {
  if (CHECKPOINT_ID) {
    const ckpt = await client.checkpoints.get(CHECKPOINT_ID);
    console.log(`  checkpoint_id=${ckpt.id}  run=${ckpt.training_run_id}`);
    return ckpt;
  }
  const all = await client.checkpoints.list();
  const mlp = all.filter((c) => c.control_model_params);
  if (!mlp.length) throw new Error("No MLP checkpoints found — run a training job first or set CHECKPOINT_ID");
  const ckpt = mlp[mlp.length - 1];
  console.log(`  checkpoint_id=${ckpt.id}  run=${ckpt.training_run_id}`);
  return ckpt;
}

function buildTestGraph(modelId, controlModel) {
  return {
    test_job_controller: {
      test_params: {
        num_timesteps: NUM_TIMESTEPS,
        control_model: controlModel,
        simulation_model: modelId,
        num_parallel_envs: 1,
      },
      compute_instance: "AWSG4dn",
      acceleration_mode: "AllOnServer",
    },
  };
}

async function runTest(client, testGraph, label) {
  const { id: testId } = await client.simulations.start(testGraph);
  console.log(`  test_id=${testId}`);

  // Each update is a TestUpdateMessage: simulation_state is [env][timestep] of
  // ModelForward { xpos, xquat }; frames (cameras) is absent here (all_cameras=false).
  const frames = [];
  for await (const frame of client.simulations.streamUpdates(testId)) {
    frames.push(frame);
    const update = frame.simulation_state[0][0];
    console.log(`    frame[${frames.length - 1}]: xpos[:3]=${update.xpos.slice(0, 3)}`);
    if (frames.length >= MAX_FRAMES) break;
  }

  if (!frames.length)                                  throw new Error(`[${label}] no frames received`);
  if (!frames[0].simulation_state[0][0].xpos.length)   throw new Error(`[${label}] empty xpos`);
  if (!frames[0].simulation_state[0][0].xquat.length)  throw new Error(`[${label}] empty xquat`);
  console.log(`  ${label}: OK (${frames.length} frames)`);

  await client.simulations.stop(testId);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

async function testVanillaMlp(client, modelId) {
  const controlModel = {
    type: "MLP",
    content: {
      model_type: { type: "New" },
      num_hidden_layers: 2,
      hidden_layer_dim: 64,
      output_dim: MLP_OUTPUT_DIM,
      input_dim: MLP_INPUT_DIM,
      input_spec: {
        all_cameras: false,
        qpos_state: true,
        qvel_state: true,
        xpos_state: false,
        xquat_state: false,
        qdof_state: false,
        contact_pairs: false,
      },
    },
  };
  await runTest(client, buildTestGraph(modelId, controlModel), "vanilla-mlp");
}

async function testCheckpointMlp(client, modelId) {
  const ckpt = await fetchMlpCheckpoint(client);
  const params = ckpt.control_model_params ?? {
    num_hidden_layers: 2,
    hidden_layer_dim: 64,
    output_dim: MLP_OUTPUT_DIM,
    input_dim: MLP_INPUT_DIM,
  };
  const controlModel = {
    type: "MLP",
    content: {
      ...params,
      model_type: {
        type: "Checkpoint",
        content: { type: "Checkpoint", content: ckpt.id },
      },
      input_spec: {
        all_cameras: false,
        qpos_state: true,
        qvel_state: true,
        xpos_state: false,
        xquat_state: false,
        qdof_state: false,
        contact_pairs: false,
      },
    },
  };
  await runTest(client, buildTestGraph(modelId, controlModel), "checkpoint-mlp");
}

// ── Entry point ───────────────────────────────────────────────────────────────

async function main() {
  const client = new LagrangianClient({ baseUrl: BASE_URL });

  console.log("=== Uploading model ===");
  const modelId = await uploadModel(client);

  console.log("\n=== Test 1: Local testing | Vanilla MLP (random weights) ===");
  await testVanillaMlp(client, modelId);

  console.log("\n=== Test 2: Local testing | MLP from checkpoint ===");
  await testCheckpointMlp(client, modelId);

  console.log("\n=== All tests passed ===");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
