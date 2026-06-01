import { streamSSE } from "../sse.js";
import type { SimulUpdateMessage, SimulationParams } from "../types.js";

export class SimulationsResource {
  constructor(
    private readonly baseUrl: string,
    private readonly fetch: (
      url: string,
      init?: RequestInit,
    ) => Promise<Response>,
  ) {}

  async start(params: SimulationParams): Promise<{ id: string }> {
    const res = await this.fetch(
      `${this.baseUrl}/testing/simulations`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ simulation_params: params }),
      },
    );
    return res.json() as Promise<{ id: string }>;
  }

  async stop(simulationId: string): Promise<{ id: string }> {
    const res = await this.fetch(
      `${this.baseUrl}/testing/simulations/${simulationId}/stop`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      },
    );
    return res.json() as Promise<{ id: string }>;
  }

  streamUpdates(simulationId: string): AsyncIterable<SimulUpdateMessage> {
    const url = `${this.baseUrl}/testing/simulations/${simulationId}/updates`;
    return streamSSE<SimulUpdateMessage>(url);
  }
}
