/**
 * Shared helpers for the direct API smoke scripts.
 * Node >= 22 and Python 3 are required; no SDK is used.
 * Set ACCESS_KEY and MODEL_PATH (except for dataset tests).
 * PROJECT_ID reuses a project; otherwise a project is created.
 * TEST_DRY_RUN=1 prints request fixtures without using services.
 * TIMEOUT_SECONDS bounds the entire workload, including startup.
 * Recorded JSONL and PNGs go to tmp/tests/test_run_assets/.
 * Large ZIPs may need ROCKET_LIMITS='{json=128MiB}' on the app.
 */
import { spawnSync } from "node:child_process";
import { appendFile, mkdir, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const dryRun = process.env.TEST_DRY_RUN === "1";
const base = new URL(
  `${(
    process.env.BASE_URL ?? "http://localhost:8000/api"
  ).replace(/\/+$/, "")}/`,
);
const artifactRoot = resolve(
  process.env.TEST_ARTIFACT_DIR ??
    fileURLToPath(
      new URL("../tmp/tests/test_run_assets/", import.meta.url),
    ),
);
const groups = [
  "logs",
  "metrics",
  "checkpoints",
  "simulation_models",
  "camera_frames",
  "model_states",
];

export function positiveInteger(name, fallback) {
  const value = Number(process.env[name] ?? fallback);
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error(`${name} must be a positive integer`);
  }
  return value;
}

function endpoint(path, query = {}) {
  const url = new URL(path, base);
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined && value !== null) {
      for (const item of Array.isArray(value) ? value : [value]) {
        url.searchParams.append(key, String(item));
      }
    }
  }
  return url;
}

export async function apiRequest(
  path,
  { method = "GET", body, query, accepted = [200] } = {},
) {
  const key = process.env.ACCESS_KEY;
  if (!key) throw new Error("ACCESS_KEY is required");
  const response = await fetch(endpoint(path, query), {
    method,
    headers: {
      Authorization: `Bearer ${key}`,
      ...(body === undefined
        ? {}
        : {
            "Content-Type": "application/json",
          }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(
      positiveInteger("HTTP_TIMEOUT_SECONDS", 300) * 1000,
    ),
  });
  if (!accepted.includes(response.status)) {
    const detail = (await response.text()).slice(0, 1500);
    const hint =
      response.status === 413
        ? " Increase the app's JSON body limit for model uploads."
        : "";
    throw new Error(
      `${method} ${path}: ${response.status} ${detail}${hint}`,
    );
  }
  return response.status === 200 ? response.json() : null;
}

export async function prepareModel(label, needsModel = true) {
  if (dryRun) {
    return {
      projectId: "project-fixture",
      modelId: "model-fixture",
    };
  }
  if (needsModel && !process.env.MODEL_PATH) {
    throw new Error("MODEL_PATH is required");
  }
  const projectId =
    process.env.PROJECT_ID ??
    (
      await apiRequest("create_project", {
        method: "POST",
        body: { name: label },
      })
    ).project_id;
  if (!projectId) throw new Error("Missing project_id");
  console.log(`project_id=${projectId}`);
  if (!needsModel) return { projectId, modelId: null };
  const modelPath = resolve(process.env.MODEL_PATH);
  // Keep all referenced XML includes, meshes and textures.
  const archive = spawnSync(
    process.env.PYTHON_BIN ?? "python3",
    [
      "-c",
      [
        "import io, pathlib, sys, zipfile",
        "source = pathlib.Path(sys.argv[1]).resolve(strict=True)",
        "assert source.is_file(), 'MODEL_PATH must be a file'",
        "output = io.BytesIO()",
        "compression = zipfile.ZIP_DEFLATED",
        "with zipfile.ZipFile(output, 'w', compression) as z:",
        " for path in sorted(source.parent.rglob('*')):",
        "  if path.is_file():",
        "   name = path.relative_to(source.parent).as_posix()",
        "   z.write(path, name)",
        "sys.stdout.buffer.write(output.getvalue())",
      ].join("\n"),
      modelPath,
    ],
    { maxBuffer: 512 * 1024 * 1024 },
  );
  if (archive.error) throw archive.error;
  if (archive.status !== 0) {
    throw new Error(`Model packaging failed: ${archive.stderr}`);
  }
  const { model_id: modelId } = await apiRequest("save_model", {
    method: "POST",
    body: {
      project_id: projectId,
      name: basename(modelPath),
      model_type: "mjcf",
      file_type: "zip",
      base_file_name: basename(modelPath),
      file_contents: Array.from(archive.stdout),
    },
  });
  if (!modelId) throw new Error("Missing model_id");
  console.log(`model_id=${modelId}`);
  return { projectId, modelId };
}

export async function findMlpCheckpoint(projectId) {
  if (dryRun) {
    return {
      checkpoint_id: "checkpoint-fixture",
      model_type: "policy",
      model_spec: {
        type: "MLP",
        content: {
          num_hidden_layers: 2,
          hidden_layer_dim: 64,
          input_dim: 55,
          output_dim: 16,
        },
      },
    };
  }
  const { checkpoints } = await apiRequest("list_checkpoints", {
    query: { project_id: projectId },
  });
  const checkpoint = checkpoints.find(
    (item) =>
      (!process.env.CHECKPOINT_ID ||
        item.checkpoint_id === process.env.CHECKPOINT_ID) &&
      item.model_type === "policy" &&
      item.model_spec.type === "MLP" &&
      item.model_spec.content.input_dim === 55 &&
      item.model_spec.content.output_dim === 16,
  );
  if (!checkpoint) {
    throw new Error(
      "No policy checkpoint with 55 inputs and 16 outputs. " +
        "Run the training smoke test first or set CHECKPOINT_ID.",
    );
  }
  return checkpoint;
}

export function zeroReward(inputSpec) {
  return {
    language: "Python",
    input_spec: inputSpec,
    reward_function_code:
      "def reward(state_pre, state_post):\n    return 0.0\n",
  };
}

export function jointInput(
  baseBody = null,
  items = ["QPos", "QVel"],
) {
  return {
    type: "StandardInputSpec",
    content: {
      base_body_name: baseBody,
      states: items.map((type) => ({
        type: "InputStateControllable",
        content: {
          rebase_body_tree: null,
          excluded_body_trees: [],
          state_item: { type },
        },
      })),
    },
  };
}

export function jointOutput() {
  return {
    type: "ProprioceptiveOutputSpec",
    content: {
      base_body_name: "base_link",
      state: {
        is_delta: false,
        rebase_body: "base_link",
        state_item: { type: "QPos" },
      },
    },
  };
}

export function vlaControl(baseBody, eefBody, gripperBody) {
  const input = jointInput(baseBody, ["QPos"]);
  input.content.states.push({
    type: "InputStateSensors",
    content: "Cameras",
  });
  return {
    model_type: {
      type: "Checkpoint",
      content: {
        type: "Pretrained",
        content: "Pi0",
      },
    },
    model: { type: "VLA" },
    input_spec: input,
    output_spec: {
      type: "EEFOutputSpec",
      content: {
        base_body_name: baseBody,
        eef_state: {
          is_delta: true,
          rebase_body: eefBody,
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
          rebase_body: gripperBody,
          state_item: { type: "QPos" },
        },
      },
    },
  };
}

async function* streamUpdates(workloadId, deadline) {
  const { ticket } = await apiRequest(
    "get_workload_live_ticket",
    {
      method: "POST",
      query: { workload_id: workloadId },
    },
  );
  const url = endpoint("get_workload_live_updates", {
    workload_id: workloadId,
    ticket,
  });
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(url);
  const queue = [];
  let wake;
  let closed = false;
  let failure;
  const notify = () => {
    wake?.();
    wake = undefined;
  };
  const timer = setTimeout(
    () => {
      failure = new Error(`Workload ${workloadId} timed out`);
      socket.close();
      notify();
    },
    Math.max(1, deadline - Date.now()),
  );
  socket.addEventListener("message", (event) => {
    try {
      if (typeof event.data !== "string") {
        throw new Error("Expected JSON text on the WebSocket");
      }
      queue.push(JSON.parse(event.data));
    } catch (error) {
      failure = error;
    }
    notify();
  });
  socket.addEventListener("error", () => {
    failure = new Error(`WebSocket failed for ${workloadId}`);
    notify();
  });
  socket.addEventListener("close", () => {
    closed = true;
    notify();
  });
  try {
    while (true) {
      if (failure) throw failure;
      if (queue.length) yield queue.shift();
      else if (closed) return;
      else
        await new Promise((resolveWake) => {
          wake = resolveWake;
        });
    }
  } finally {
    clearTimeout(timer);
    socket.close();
  }
}

function recordKey(group, record) {
  const id = {
    logs: "log_id",
    checkpoints: "checkpoint_id",
    simulation_models: "simulation_model_id",
    camera_frames: "camera_frame_id",
    model_states: "sim_model_state_entry_id",
  }[group];
  const value = id
    ? record[id]
    : JSON.stringify([
        record.created_at,
        record.metric_name,
        record.workload_structure_state_values,
        record.metric_values,
      ]);
  if (!value) throw new Error(`Missing identifier in ${group}`);
  return `${group}/${value}`;
}

function coordinate(record) {
  const names = record.workload_structure_index_names;
  const values = record.workload_structure_state;
  if (
    !Array.isArray(names) ||
    names.length !== values?.length ||
    values.some(
      (value) => !Number.isSafeInteger(value) || value < 0,
    )
  ) {
    throw new Error("Invalid workload coordinates or LUT names");
  }
  return Object.fromEntries(
    names.map((name, i) => [name, values[i]]),
  );
}

function coordinateKey(value) {
  return JSON.stringify([
    value.task_index ?? 0,
    value.rollout ?? 0,
    value.trajectory ?? value.rollout_trajectory,
    value.trajectory_timestep ??
      value.rollout_trajectory_timestep,
  ]);
}

function validateWorld(world) {
  if (!world?.bodies?.length) throw new Error("Empty WorldBody");
  const pending = [...world.bodies];
  for (const body of pending) {
    for (const [field, length] of [
      ["pos", 3],
      ["quat", 4],
    ]) {
      if (
        body[field]?.length !== length ||
        body[field].some((value) => !Number.isFinite(value))
      ) {
        throw new Error(`Invalid WorldBody ${field}`);
      }
    }
    pending.push(...(body.bodies ?? []));
  }
}

async function saveFrame(frame, directory, label, timestamp) {
  const url = new URL(frame.uri_path, base);
  if (
    url.origin !== base.origin ||
    !url.pathname.startsWith(`${base.pathname}assets/download/`)
  ) {
    throw new Error("Expected an app-owned camera download link");
  }
  const response = await fetch(url, {
    signal: AbortSignal.timeout(60_000),
  });
  if (!response.ok) {
    throw new Error(`Frame download failed: ${response.status}`);
  }
  const data = Buffer.from(await response.arrayBuffer());
  const signature = Buffer.from([
    137, 80, 78, 71, 13, 10, 26, 10,
  ]);
  if (
    data.length < 24 ||
    !data.subarray(0, 8).equals(signature) ||
    data.toString("ascii", 12, 16) !== "IHDR" ||
    !data.readUInt32BE(16) ||
    !data.readUInt32BE(20)
  ) {
    throw new Error("Downloaded camera asset is not a valid PNG");
  }
  const parts = [
    timestamp,
    label,
    coordinateKey(coordinate(frame)),
    frame.simulation_model_id,
    frame.camera_id,
  ];
  const filename =
    parts
      .map((part) =>
        String(part).replace(/[^a-zA-Z0-9_.-]/g, "_"),
      )
      .join("__") + ".png";
  await writeFile(join(directory, filename), data, {
    flag: "wx",
  });
}

async function verifyHistory(workloadId, seen) {
  const historical = new Set();
  const cursors = new Set();
  let cursor;
  do {
    const page = await apiRequest("get_workload_updates", {
      query: { workload_id: workloadId, limit: 37, cursor },
    });
    if (page.workload_id !== workloadId) {
      throw new Error("History belongs to another workload");
    }
    let count = 0;
    for (const update of page.updates) {
      for (const group of groups) {
        count += update[group].length;
        for (const record of update[group]) {
          historical.add(recordKey(group, record));
        }
      }
    }
    if (count > 37) throw new Error("History page exceeds limit");
    cursor = page.next_cursor;
    if (cursor && cursors.has(cursor)) {
      throw new Error("History cursor did not advance");
    }
    if (cursor) cursors.add(cursor);
  } while (cursor);
  if (
    historical.size !== seen.size ||
    [...historical].some((key) => !seen.has(key))
  ) {
    throw new Error(
      "Persisted history differs from live records",
    );
  }
}

export async function runWorkload(
  kind,
  projectId,
  graph,
  label,
  expected,
) {
  const request = {
    project_id: projectId,
    [`${kind}_graph`]: graph,
    name: label,
    description: `API smoke test: ${label}`,
    tags: ["smoke-test"],
  };
  if (dryRun) {
    console.log(
      JSON.stringify({ dry_run: true, request, expected }),
    );
    return;
  }
  const maxUpdates = positiveInteger("MAX_UPDATES", 10000);
  const deadline =
    Date.now() + positiveInteger("TIMEOUT_SECONDS", 3600) * 1000;
  const { workload_id: workloadId } = await apiRequest(
    kind === "training" ? "start_training" : "start_test",
    { method: "POST", body: request },
  );
  if (!workloadId) throw new Error("Missing workload_id");
  const timestamp = new Date()
    .toISOString()
    .replace(/[-:.]/g, "");
  const directory = join(
    artifactRoot,
    `${timestamp}__${workloadId}`,
  );
  const seen = new Set();
  const states = new Map();
  const cameras = new Map();
  const checkpoints = new Set();
  let metricCount = 0;
  let stopped = false;
  let messages = 0;
  try {
    await mkdir(directory, { recursive: true });
    console.log(
      `workload_id=${workloadId} artifacts=${directory}`,
    );
    for await (const message of streamUpdates(
      workloadId,
      deadline,
    )) {
      if (Date.now() > deadline)
        throw new Error("Workload timed out");
      await appendFile(
        join(directory, "updates.jsonl"),
        `${JSON.stringify(message)}\n`,
      );
      if (message.WorkloadStoppedMessage) {
        const terminal = message.WorkloadStoppedMessage;
        if (terminal.workload_id !== workloadId) {
          throw new Error(
            "Stop event belongs to another workload",
          );
        }
        stopped = true;
        if (terminal.reason !== "CompletedSuccess") {
          throw new Error(
            `Workload failed: ${JSON.stringify(terminal.reason)}`,
          );
        }
        break;
      }
      if (!Array.isArray(message.WorkloadUpdatesMessage)) {
        throw new Error("Unknown workload message");
      }
      messages += 1;
      if (messages > maxUpdates) {
        throw new Error("MAX_UPDATES reached before completion");
      }
      for (const update of message.WorkloadUpdatesMessage) {
        if (update.workload_id !== workloadId) {
          throw new Error("Update belongs to another workload");
        }
        for (const group of groups) {
          for (const record of update[group]) {
            if (
              record.workload_id != null &&
              record.workload_id !== workloadId
            ) {
              throw new Error(
                "Record belongs to another workload",
              );
            }
            const key = recordKey(group, record);
            if (seen.has(key)) continue;
            seen.add(key);
            if (group === "logs")
              console.log(record.log_line_text);
            if (group === "metrics") {
              if (
                record.metric_values.length !==
                  record.workload_structure_state_values.length ||
                record.metric_values.some(
                  (v) => !Number.isFinite(v),
                )
              ) {
                throw new Error("Invalid metric batch");
              }
              metricCount += record.metric_values.length;
            }
            if (group === "checkpoints")
              checkpoints.add(record.model_type);
            if (group === "model_states") {
              validateWorld(record.state);
              const coord = coordinateKey(coordinate(record));
              if (states.has(coord)) {
                throw new Error(
                  `Repeated state coordinate ${coord}`,
                );
              }
              states.set(coord, record.simulation_model_id);
            }
            if (group === "camera_frames") {
              await saveFrame(
                record,
                directory,
                label,
                timestamp,
              );
              const camera = JSON.stringify([
                record.simulation_model_id,
                record.camera_id,
              ]);
              if (!cameras.has(camera))
                cameras.set(camera, new Set());
              const coord = coordinateKey(coordinate(record));
              if (cameras.get(camera).has(coord)) {
                throw new Error(
                  `Repeated camera coordinate ${coord}`,
                );
              }
              cameras.get(camera).add(coord);
            }
          }
        }
      }
      console.log(`update=${messages} states=${states.size}`);
    }
    if (!stopped)
      throw new Error("Socket closed before completion");
    const tasks = expected.tasks ?? [0];
    const rollouts = expected.rollouts ?? 1;
    const total =
      tasks.length *
      rollouts *
      expected.trajectories *
      expected.timesteps;
    for (const task of tasks)
      for (let rollout = 0; rollout < rollouts; rollout += 1) {
        for (
          let trajectory = 0;
          trajectory < expected.trajectories;
          trajectory += 1
        ) {
          for (
            let timestep = 0;
            timestep < expected.timesteps;
            timestep += 1
          ) {
            const key = JSON.stringify([
              task,
              rollout,
              trajectory,
              timestep,
            ]);
            if (!states.has(key))
              throw new Error(`Missing state ${key}`);
          }
        }
      }
    if (states.size !== total)
      throw new Error("Unexpected state coordinates");
    if (expected.cameras) {
      const recordedModels = new Set(
        [...cameras.keys()].map((key) => JSON.parse(key)[0]),
      );
      if (
        [...states.values()].some((id) => !recordedModels.has(id))
      ) {
        throw new Error("Missing camera recordings for a model");
      }
    }
    for (const [camera, coordinates] of cameras) {
      const [modelId] = JSON.parse(camera);
      const required = [...states].filter(
        ([, id]) => id === modelId,
      );
      if (
        coordinates.size !== required.length ||
        required.some(([coord]) => !coordinates.has(coord))
      ) {
        throw new Error(`Incomplete camera coverage: ${camera}`);
      }
    }
    if (
      kind === "training" &&
      (!metricCount ||
        !checkpoints.has("policy") ||
        !checkpoints.has("value"))
    ) {
      throw new Error(
        "Training metrics or scheduled checkpoints are missing",
      );
    }
    if (kind === "training") {
      const stored = await apiRequest(
        "get_workload_checkpoints",
        {
          query: { workload_ids: [workloadId] },
        },
      );
      const roles = new Set(
        stored.checkpoints.map((c) => c.model_type),
      );
      if (!roles.has("policy") || !roles.has("value")) {
        throw new Error("Missing persisted training checkpoints");
      }
    }
    await verifyHistory(workloadId, seen);
    const { workloads } = await apiRequest("list_workloads", {
      query: { project_id: projectId },
    });
    if (
      workloads.find((w) => w.workload_id === workloadId)
        ?.status !== "completed"
    )
      throw new Error("Final workload status is not completed");
    console.log(
      `PASS ${label}: ${states.size} states, ${messages} updates`,
    );
  } finally {
    if (!stopped) {
      await apiRequest("stop_workload", {
        method: "POST",
        body: { workload_id: workloadId },
        accepted: [202, 409],
      }).catch((error) =>
        console.error(`Stop cleanup: ${error.message}`),
      );
    }
  }
}
