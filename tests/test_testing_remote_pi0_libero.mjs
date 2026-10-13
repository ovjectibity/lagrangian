/**
 * Remote MJX testing workload test using Pi-0 on LIBERO tasks.
 *
 * Requires a GPU worker with the LIBERO dataset dependency available.
 *
 * Usage:
 *   node tests/test_testing_remote_pi0_libero.mjs
 *   BASE_URL=http://localhost:8000/api \
 *     LIBERO_DATASET=LiberoSpatial TASK_INDICES=0,1 \
 *     NUM_ENVS=2 NUM_TIMESTEPS=30 \
 *     node tests/test_testing_remote_pi0_libero.mjs
 *
 * Requires: node >= 18, sdk built (cd sdk && npm run build).
 */

import { LagrangianClient } from "../sdk/dist/index.js";

const BASE_URL = process.env.BASE_URL ?? "http://localhost:8000/api";
const LIBERO_DATASET = process.env.LIBERO_DATASET ?? "LiberoSpatial";
const TASK_INDICES = (process.env.TASK_INDICES ?? "0")
  .split(",")
  .map((value) => Number.parseInt(value, 10));
const NUM_ENVS = Number.parseInt(process.env.NUM_ENVS ?? "1", 10);
const NUM_TIMESTEPS = Number.parseInt(process.env.NUM_TIMESTEPS ?? "30", 10);
const MAX_UPDATES = Number.parseInt(process.env.MAX_UPDATES ?? "100", 10);

if (
  !TASK_INDICES.length
  || TASK_INDICES.some((taskIndex) => !Number.isInteger(taskIndex))
) {
  throw new Error("TASK_INDICES must be a comma-separated list of integers");
}

function buildTestGraph() {
  return {
    test_job_controller: {
      test_method: {
        type: "Dataset",
        content: {
          dataset: {
            type: "Libero",
            content: {
              tasks_index: TASK_INDICES,
              libero_dataset_type: LIBERO_DATASET,
              num_envs_per_task: NUM_ENVS,
              num_timesteps: NUM_TIMESTEPS,
            },
          },
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
                contact_pairs: false,
              },
              output: "TargetPositions",
            },
          },
        },
      },
      compute_instance: "AWSG4dn",
      acceleration_mode: "MLGPUSimulationGPU",
      // acceleration_mode: "AllOnServer",
    },
  };
}

function validateCameraFrames(taskIndex, cameraFrames) {
  if (!cameraFrames) {
    throw new Error(`task ${taskIndex}: expected camera frames`);
  }
  if (!cameraFrames.frames.length) {
    throw new Error(`task ${taskIndex}: camera frame buffer is empty`);
  }
  const expectedLength =
    cameraFrames.nenv
    * cameraFrames.t
    * cameraFrames.ncam
    * cameraFrames.h
    * cameraFrames.w
    * 3;
  if (cameraFrames.frames.length !== expectedLength) {
    throw new Error(
      `task ${taskIndex}: camera frame length ${cameraFrames.frames.length} `
      + `!= nenv*t*ncam*h*w*3 (${expectedLength})`,
    );
  }
}

async function main() {
  const client = new LagrangianClient({ baseUrl: BASE_URL });
  const { id: testId } = await client.simulations.start(buildTestGraph());
  console.log(`test_id=${testId}`);

  const observedTasks = new Set();
  let updateCount = 0;
  try {
    for await (const frame of client.simulations.streamUpdates(testId)) {
      updateCount += 1;
      for (const taskIndex of TASK_INDICES) {
        const taskStates = frame.simulation_state[taskIndex];
        if (!taskStates) {
          continue;
        }
        const firstUpdate = taskStates[0]?.[0];
        if (!firstUpdate?.xpos.length || !firstUpdate?.xquat.length) {
          throw new Error(`task ${taskIndex}: empty body pose update`);
        }
        validateCameraFrames(taskIndex, frame.frames[taskIndex]);
        observedTasks.add(taskIndex);
        console.log(
          `update=${updateCount} task=${taskIndex} `
          + `xpos[:3]=${firstUpdate.xpos.slice(0, 3)}`,
        );
      }
      if (observedTasks.size === TASK_INDICES.length || updateCount >= MAX_UPDATES) {
        break;
      }
    }
  } finally {
    await client.simulations.stop(testId);
  }

  const missingTasks = TASK_INDICES.filter(
    (taskIndex) => !observedTasks.has(taskIndex),
  );
  if (missingTasks.length) {
    throw new Error(
      `no testing updates received for task indices: ${missingTasks.join(", ")}`,
    );
  }
  console.log(`LIBERO Pi-0 test: OK (${updateCount} updates)`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
