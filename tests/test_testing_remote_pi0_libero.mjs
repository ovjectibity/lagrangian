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
 * Requires: node >= 18, npm install --prefix tests,
 * sdk built (cd sdk && npm run build).
 * Camera PNGs are saved under <project>/tmp/tests/test_run_assets,
 * with a separate timestamped directory for each test run.
 */

import { createWriteStream } from "node:fs";
import { mkdir } from "node:fs/promises";
import { basename, join } from "node:path";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";
import { PNG } from "pngjs";

import { LagrangianClient } from "../sdk/dist/index.js";

const TEST_NAME = basename(fileURLToPath(import.meta.url), ".mjs");
const ASSETS_ROOT = fileURLToPath(
  new URL("../tmp/tests/test_run_assets/", import.meta.url),
);
const BASE_URL = process.env.BASE_URL ?? "http://localhost:8000/api";
const LIBERO_DATASET = process.env.LIBERO_DATASET ?? "LiberoSpatial";
const TASK_INDICES = (process.env.TASK_INDICES ?? "0")
  .split(",")
  .map((value) => Number.parseInt(value, 10));
const NUM_ENVS = Number.parseInt(process.env.NUM_ENVS ?? "1", 10);
const NUM_TIMESTEPS = Number.parseInt(process.env.NUM_TIMESTEPS ?? "20", 10);
const MAX_UPDATES = Number.parseInt(process.env.MAX_UPDATES ?? "100", 10);
const ROBOT_BASE_BODY = process.env.ROBOT_BASE_BODY ?? "robot0_base";
const EEF_BODY = process.env.EEF_BODY ?? "gripper0_right_eef";
// Command the driven finger; the equality couples the opposing jaw.
const GRIPPER_BODY = process.env.GRIPPER_BODY
  ?? "gripper0_right_leftfinger";

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
            model: {
              type: "VLA",
              content: {
                model_type: {
                  type: "Checkpoint",
                  content: { type: "Pretrained", content: "Pi0" },
                },
              },
            },
            input_spec: {
              type: "StandardInputSpec",
              content: {
                base_body_name: ROBOT_BASE_BODY,
                states: [
                  {
                    type: "InputStateControllable",
                    content: {
                      rebase_body_tree: null,
                      excluded_body_trees: [],
                      state_item: { type: "QPos" },
                    },
                  },
                  { type: "InputStateSensors", content: "Cameras" },
                ],
              },
            },
            output_spec: {
              type: "EEFOutputSpec",
              content: {
                base_body_name: ROBOT_BASE_BODY,
                eef_state: {
                  is_delta: true,
                  rebase_body: EEF_BODY,
                  state_item: {
                    type: "XPose",
                    content: {
                      frame_spec: "WorldFrame",
                      orientation_spec: "AxisAngle",
                    },
                  },
                },
                gripper: {
                  is_delta: true,
                  rebase_body: GRIPPER_BODY,
                  state_item: { type: "QPos" },
                },
              },
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

async function saveCameraFrames(
  outputDirectory,
  runTimestamp,
  taskIndex,
  cameraFrames,
  firstTimestep,
) {
  const { nenv, t, ncam, h, w } = cameraFrames;
  const pixels = Buffer.from(cameraFrames.frames);
  const imageBytes = h * w * 3;
  const prefix = [
    runTimestamp,
    TEST_NAME,
    LIBERO_DATASET,
    `task${String(taskIndex).padStart(3, "0")}`,
  ].join("__");
  let savedImages = 0;

  for (let envIndex = 0; envIndex < nenv; envIndex += 1) {
    for (let localStep = 0; localStep < t; localStep += 1) {
      for (let camera = 0; camera < ncam; camera += 1) {
        const offset =
          ((envIndex * t + localStep) * ncam + camera)
          * imageBytes;
        const timestep = firstTimestep + localStep;
        const filename = [
          prefix,
          `step${String(timestep).padStart(6, "0")}`,
          `env${String(envIndex).padStart(2, "0")}`,
          `cam${String(camera).padStart(2, "0")}`,
        ].join("__") + ".png";
        const image = new PNG({
          width: w,
          height: h,
          colorType: 2,
          inputColorType: 2,
          inputHasAlpha: false,
        });
        image.data = pixels.subarray(offset, offset + imageBytes);
        await pipeline(
          image.pack(),
          createWriteStream(
            join(outputDirectory, filename),
            { flags: "wx" },
          ),
        );
        savedImages += 1;
      }
    }
  }
  return savedImages;
}

async function main() {
  const runTimestamp =
    new Date().toISOString().replace(/[-:.]/g, "");
  const client = new LagrangianClient({ baseUrl: BASE_URL });
  const { id: testId } = await client.simulations.start(buildTestGraph());
  console.log(`test_id=${testId}`);
  const outputDirectory = join(
    ASSETS_ROOT, `${runTimestamp}__${testId}`,
  );

  const receivedTimesteps = new Map(
    TASK_INDICES.map((taskIndex) => [taskIndex, 0]),
  );
  let updateCount = 0;
  let savedImages = 0;
  try {
    await mkdir(outputDirectory, { recursive: true });
    console.log(`test_run_assets=${outputDirectory}`);
    for await (const frame of client.simulations.streamUpdates(testId)) {
      updateCount += 1;
      console.log(
        `update=${updateCount} `
        + `received_at=${new Date().toISOString()}`,
      );
      for (const taskIndex of TASK_INDICES) {
        const taskStates = frame.simulation_state[taskIndex];
        if (!taskStates) {
          continue;
        }
        const firstUpdate = taskStates[0]?.[0];
        if (!firstUpdate?.xpos.length || !firstUpdate?.xquat.length) {
          throw new Error(`task ${taskIndex}: empty body pose update`);
        }
        const chunkTimesteps = taskStates[0].length;
        if (
          taskStates.length !== NUM_ENVS
          || taskStates.some(
            (states) => states.length !== chunkTimesteps,
          )
        ) {
          throw new Error(
            `task ${taskIndex}: inconsistent environment updates`,
          );
        }
        const cameraFrames = frame.frames[taskIndex];
        validateCameraFrames(taskIndex, cameraFrames);
        if (
          cameraFrames.nenv !== taskStates.length
          || cameraFrames.t !== chunkTimesteps
        ) {
          throw new Error(
            `task ${taskIndex}: camera/state dimensions differ`,
          );
        }
        const firstTimestep = receivedTimesteps.get(taskIndex);
        const chunkImages = await saveCameraFrames(
          outputDirectory, runTimestamp, taskIndex,
          cameraFrames, firstTimestep,
        );
        savedImages += chunkImages;
        const totalTimesteps =
          firstTimestep + chunkTimesteps;
        receivedTimesteps.set(taskIndex, totalTimesteps);
        console.log(
          `update=${updateCount} task=${taskIndex} `
          + `timesteps=${totalTimesteps}/${NUM_TIMESTEPS} `
          + `images_saved=${chunkImages} `
          + `xpos[:3]=${firstUpdate.xpos.slice(0, 3)}`,
        );
      }
      const complete = TASK_INDICES.every(
        (taskIndex) =>
          receivedTimesteps.get(taskIndex) >= NUM_TIMESTEPS,
      );
      if (complete || updateCount >= MAX_UPDATES) {
        break;
      }
    }
  } finally {
    await client.simulations.stop(testId);
  }

  const incompleteTasks = TASK_INDICES.filter(
    (taskIndex) =>
      receivedTimesteps.get(taskIndex) < NUM_TIMESTEPS,
  );
  if (incompleteTasks.length) {
    const progress = incompleteTasks.map(
      (taskIndex) =>
        `task ${taskIndex}: ${receivedTimesteps.get(taskIndex)}`
        + `/${NUM_TIMESTEPS} timesteps`,
    );
    throw new Error(
      `incomplete testing updates: ${progress.join(", ")}`,
    );
  }
  console.log(`LIBERO Pi-0 test: OK (${updateCount} updates)`);
  console.log(`Saved ${savedImages} PNGs to ${outputDirectory}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
