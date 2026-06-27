import { util, configure } from "protobufjs/minimal";
import { streamBinaryWS } from "../ws.js";
import { lagrangian } from "../proto/testing.js";
import type {
  CameraFrames,
  ModelForward,
  TestGraph,
  TestUpdateMessage,
} from "../types.js";

const { TestingStreamMessage } = lagrangian.testing;

// Decode 64-bit protobuf ints to plain JS numbers (the dims/nbody are small).
(util as unknown as { Long: unknown }).Long = null;
configure();

// WS binary frames for /testing/<id>/updates carry a protobuf
// `lagrangian.testing.TestingStreamMessage` (see
// docs/user/specs/testing_updates.proto). Decode it and map to the typeshare
// `TestUpdateMessage` so the public SDK type is unchanged for end users.
function decodeSimulFrame(buf: ArrayBuffer): TestUpdateMessage {
  const env = TestingStreamMessage.decode(new Uint8Array(buf));
  const update = env.testUpdate;
  if (!update) {
    return { id: "", simulation_state: [] };
  }

  // EnvUpdates[] -> ModelForward[][]; drop the proto-only `nbody` field.
  const simulation_state: ModelForward[][] = (
    update.simulationState ?? []
  ).map((envUpdates) =>
    (envUpdates.updates ?? []).map((mf) => ({
      xpos: mf.xpos ?? [],
      xquat: mf.xquat ?? [],
    })),
  );

  const result: TestUpdateMessage = {
    id: update.id ?? "",
    simulation_state,
  };

  if (update.frames) {
    const f = update.frames;
    const frames: CameraFrames = {
      frames: Array.from(f.frames ?? new Uint8Array()),
      nenv: Number(f.nenv ?? 0),
      ncam: Number(f.ncam ?? 0),
      t: Number(f.t ?? 0),
      h: Number(f.h ?? 0),
      w: Number(f.w ?? 0),
    };
    result.frames = frames;
  }

  return result;
}

export class SimulationsResource {
  constructor(
    private readonly baseUrl: string,
    private readonly fetch: (
      url: string,
      init?: RequestInit,
    ) => Promise<Response>,
  ) {}

  async start(graph: TestGraph): Promise<{ id: string }> {
    const res = await this.fetch(
      `${this.baseUrl}/testing/start`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ test_graph: graph }),
      },
    );
    return res.json() as Promise<{ id: string }>;
  }

  async stop(testId: string): Promise<{ id: string }> {
    const res = await this.fetch(
      `${this.baseUrl}/testing/${testId}/stop`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      },
    );
    return res.json() as Promise<{ id: string }>;
  }

  async *streamUpdates(testId: string): AsyncGenerator<TestUpdateMessage> {
    const url = `${this.baseUrl}/testing/${testId}/updates`;
    for await (const buf of streamBinaryWS(url)) {
      yield decodeSimulFrame(buf);
    }
  }
}
