import { streamWS } from "../ws.js";
import type {
  CheckpointReference,
  ClientMessage,
  TrainingExperiment,
  TrainingExperimentRun,
  TrainingGraph,
} from "../types.js";

export class ExperimentsResource {
  constructor(
    private readonly baseUrl: string,
    private readonly fetch: (
      url: string,
      init?: RequestInit,
    ) => Promise<Response>,
  ) {}

  async list(): Promise<TrainingExperiment[]> {
    const res = await this.fetch(
      `${this.baseUrl}/training/experiments`,
    );
    const body = await res.json();
    return (body as { experiments: TrainingExperiment[] }).experiments;
  }

  async create(
    body?: { name?: string },
  ): Promise<TrainingExperiment> {
    const res = await this.fetch(
      `${this.baseUrl}/training/experiments`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body ?? {}),
      },
    );
    const json = await res.json();
    return (json as { experiment: TrainingExperiment }).experiment;
  }

  async listRuns(
    experimentId: string,
  ): Promise<TrainingExperimentRun[]> {
    const res = await this.fetch(
      `${this.baseUrl}/training/experiments/${experimentId}/runs`,
    );
    const body = await res.json();
    return (body as { runs: TrainingExperimentRun[] }).runs;
  }

  async createRun(
    experimentId: string,
  ): Promise<TrainingExperimentRun> {
    const res = await this.fetch(
      `${this.baseUrl}/training/experiments/${experimentId}/runs`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      },
    );
    const body = await res.json();
    return (body as { run: TrainingExperimentRun }).run;
  }

  async getRun(
    experimentId: string,
    runId: string,
  ): Promise<TrainingExperimentRun> {
    const res = await this.fetch(
      `${this.baseUrl}/training/experiments/${experimentId}/runs/${runId}`,
    );
    const body = await res.json();
    return (body as { run: TrainingExperimentRun }).run;
  }

  async getRunCheckpoints(
    experimentId: string,
    runId: string,
  ): Promise<CheckpointReference[]> {
    const res = await this.fetch(
      `${this.baseUrl}/training/experiments/${experimentId}/runs/${runId}/checkpoints`,
    );
    const body = await res.json();
    return (body as { checkpoints: CheckpointReference[] }).checkpoints;
  }

  async startTraining(
    experimentId: string,
    runId: string,
    graph: TrainingGraph,
  ): Promise<{ id: string }> {
    const res = await this.fetch(
      `${this.baseUrl}/training/experiments/${experimentId}/runs/${runId}/start`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ training_graph: graph }),
      },
    );
    return res.json() as Promise<{ id: string }>;
  }

  async stopTraining(
    experimentId: string,
    runId: string,
  ): Promise<{ id: string }> {
    const res = await this.fetch(
      `${this.baseUrl}/training/experiments/${experimentId}/runs/${runId}/stop`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      },
    );
    return res.json() as Promise<{ id: string }>;
  }

  streamTrainingUpdates(
    experimentId: string,
    runId: string,
  ): AsyncIterable<ClientMessage> {
    const url = `${this.baseUrl}/training/experiments/${experimentId}/runs/${runId}/updates`;
    return streamWS<ClientMessage>(url);
  }
}
