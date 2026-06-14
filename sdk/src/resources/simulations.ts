import { streamBinarySSE } from "../sse.js";
import type { TestGraph, TestUpdateMessage } from "../types.js";

// SSE frames for /testing/<id>/updates are base64-encoded binary.
// Layout: [1 msg_type] [2 id_len LE] [id_len id] [2 nbody LE]
//         [nbody*3 f64 LE xpos] [nbody*4 f64 LE xquat]
function decodeSimulFrame(b64: string): TestUpdateMessage {
  const raw = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  let off = 1; // skip msg_type byte
  const idLen = view.getUint16(off, true); off += 2;
  const id = new TextDecoder().decode(raw.slice(off, off + idLen)); off += idLen;
  const nbody = view.getUint16(off, true); off += 2;
  const xpos: number[] = [];
  for (let i = 0; i < nbody * 3; i++) { xpos.push(view.getFloat64(off, true)); off += 8; }
  const xquat: number[] = [];
  for (let i = 0; i < nbody * 4; i++) { xquat.push(view.getFloat64(off, true)); off += 8; }
  return { id, updates: [[{ xpos, xquat }]] };
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
    for await (const b64 of streamBinarySSE(url)) {
      yield decodeSimulFrame(b64);
    }
  }
}
