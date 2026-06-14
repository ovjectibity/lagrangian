import { streamSSE } from "../sse.js";
import type { SimulUpdateMessage, TestGraph } from "../types.js";

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

  streamUpdates(testId: string): AsyncIterable<SimulUpdateMessage> {
    const url = `${this.baseUrl}/testing/${testId}/updates`;
    return streamSSE<SimulUpdateMessage>(url);
  }
}
