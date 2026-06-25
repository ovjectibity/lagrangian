/**
 * Remote (MLGPUSimulationGPU) testing workload test with the Pi-0 pretrained VLA model.
 * Requires a live GPU worker running and reachable by the server.
 *
 * Usage:
 *   MODEL_PATH=assets/trossen_vx300s/vx300s.xml node tests/test_testing_remote_pi0_trossen_arm.mjs
 *   BASE_URL=http://localhost:8000/api MODEL_PATH=... NUM_ENVS=4 NUM_TIMESTEPS=50 node tests/test_testing_remote_pi0_trossen_arm.mjs
 *
 * Requires: node >= 18, sdk built (cd sdk && npm run build)
 */

import { spawnSync } from "node:child_process";
import { readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { LagrangianClient } from "../sdk/dist/index.js";

const BASE_URL      = process.env.BASE_URL       ?? "http://localhost:8000/api";
const MODEL_PATH    = process.env.MODEL_PATH      ?? "";
// const NUM_ENVS      = parseInt(process.env.NUM_ENVS      ?? "4",  10);
const NUM_ENVS      = 1;
// const NUM_TIMESTEPS = parseInt(process.env.NUM_TIMESTEPS ?? "50", 10);
const NUM_TIMESTEPS = 30;

const MAX_FRAMES = 100;

if (!MODEL_PATH) {
  console.error("MODEL_PATH env var is required");
  process.exit(1);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

async function uploadModel(client) {
  const absModelPath = resolve(MODEL_PATH);
  const modelDir = dirname(absModelPath);
  const modelFile = basename(absModelPath);
  const zipPath = join(tmpdir(), `lagrangian-model-${Date.now()}.zip`);
  try {
    // Bundle all non-XML assets (meshes, textures, etc.) then add only the
    // target XML so the worker doesn't pick the wrong XML at the zip root.
    const r1 = spawnSync("zip", ["-r", zipPath, ".", "--exclude", "*.xml"], { cwd: modelDir });
    if (r1.error) throw r1.error;
    if (r1.status !== 0) throw new Error(`zip failed (${r1.status}): ${r1.stderr?.toString()}`);
    const r2 = spawnSync("zip", [zipPath, modelFile], { cwd: modelDir });
    if (r2.error) throw r2.error;
    if (r2.status !== 0) throw new Error(`zip failed (${r2.status}): ${r2.stderr?.toString()}`);
    const blob = new Blob([readFileSync(zipPath)], { type: "application/zip" });
    const result = await client.models.upload(blob, modelFile.replace(".xml", ".zip"));
    console.log(`  model_id=${result.id}`);
    return result.id;
  } finally {
    rmSync(zipPath, { force: true });
  }
}

function buildTestGraph(modelId) {
  return {
    test_job_controller: {
      test_params: {
        num_timesteps: NUM_TIMESTEPS,
        control_model: {
          type: "VLA",
          content: {
            model_type: {
              type: "Checkpoint",
              content: { type: "Pretrained", content: "Pi0" },
            },
            input_spec: {
              all_cameras: true,
              qpos_state: true,
              qvel_state: false,
              xpos_state: false,
              xquat_state: false,
              qdof_state: false,
              contact_pairs: false,
            },
            output: "TargetPositions",
          },
        },
        simulation_model: modelId,
        num_parallel_envs: NUM_ENVS,
      },
      compute_instance: "AWSG4dn",
      acceleration_mode: "MLGPUSimulationGPU",
    },
  };
}

async function runTest(client, testGraph) {
  const { id: testId } = await client.simulations.start(testGraph);
  console.log(`  test_id=${testId}`);

  const frames = [];
  for await (const frame of client.simulations.streamUpdates(testId)) {
    frames.push(frame);
    const update = frame.updates[0][0];
    console.log(`    frame[${frames.length - 1}]: xpos[:3]=${update.xpos.slice(0, 3)}`);
    if (frames.length >= MAX_FRAMES) break;
  }

  if (!frames.length)                      throw new Error("no frames received");
  if (!frames[0].updates[0][0].xpos.length)  throw new Error("empty xpos in first frame");
  if (!frames[0].updates[0][0].xquat.length) throw new Error("empty xquat in first frame");
  console.log(`  Pi-0 remote test: OK (${frames.length} frames)`);

  await client.simulations.stop(testId);
}

// ── Entry point ───────────────────────────────────────────────────────────────

async function main() {
  const client = new LagrangianClient({ baseUrl: BASE_URL });

  console.log("=== Uploading model ===");
  const modelId = await uploadModel(client);

  console.log(`\n=== Remote testing | Pi-0 pretrained VLA | MLGPUSimulationGPU | ${NUM_ENVS} parallel envs ===`);
  await runTest(client, buildTestGraph(modelId));

  console.log("\n=== All tests passed ===");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
