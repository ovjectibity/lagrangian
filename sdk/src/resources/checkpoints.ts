import type { CheckpointReference } from "../types.js";

export class CheckpointsResource {
  constructor(
    private readonly baseUrl: string,
    private readonly fetch: (
      url: string,
      init?: RequestInit,
    ) => Promise<Response>,
  ) {}

  async list(): Promise<CheckpointReference[]> {
    const res = await this.fetch(`${this.baseUrl}/checkpoints`);
    const body = await res.json();
    return (body as { checkpoints: CheckpointReference[] }).checkpoints;
  }

  async get(checkpointId: string): Promise<CheckpointReference> {
    const res = await this.fetch(
      `${this.baseUrl}/checkpoints/${checkpointId}`,
    );
    const body = await res.json();
    return (body as { checkpoint: CheckpointReference }).checkpoint;
  }
}
