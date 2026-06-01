import { LagrangianError } from "./errors.js";
import { CheckpointsResource } from "./resources/checkpoints.js";
import { ExperimentsResource } from "./resources/experiments.js";
import { ModelsResource } from "./resources/models.js";
import { SimulationsResource } from "./resources/simulations.js";

export interface LagrangianClientConfig {
  baseUrl: string;
  headers?: Record<string, string>;
}

export class LagrangianClient {
  readonly models: ModelsResource;
  readonly experiments: ExperimentsResource;
  readonly checkpoints: CheckpointsResource;
  readonly simulations: SimulationsResource;

  constructor(config: LagrangianClientConfig) {
    const baseUrl = config.baseUrl.replace(/\/$/, "");
    const extraHeaders = config.headers ?? {};

    const wrappedFetch = async (
      url: string,
      init?: RequestInit,
    ): Promise<Response> => {
      const res = await fetch(url, {
        ...init,
        headers: {
          ...extraHeaders,
          ...(init?.headers as Record<string, string> | undefined),
        },
      });
      if (!res.ok) {
        throw new LagrangianError(res.status, await res.text());
      }
      return res;
    };

    this.models = new ModelsResource(baseUrl, wrappedFetch);
    this.experiments = new ExperimentsResource(baseUrl, wrappedFetch);
    this.checkpoints = new CheckpointsResource(baseUrl, wrappedFetch);
    this.simulations = new SimulationsResource(baseUrl, wrappedFetch);
  }
}
