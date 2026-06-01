import type { ModelReference, WorldBody } from "../types.js";

export class ModelsResource {
  constructor(
    private readonly baseUrl: string,
    private readonly fetch: (
      url: string,
      init?: RequestInit,
    ) => Promise<Response>,
  ) {}

  async list(): Promise<ModelReference[]> {
    const res = await this.fetch(`${this.baseUrl}/models`);
    const body = await res.json();
    return (body as { models: ModelReference[] }).models;
  }

  async getDefault(): Promise<{ id: string; model: WorldBody }> {
    const res = await this.fetch(`${this.baseUrl}/models/default`);
    return res.json() as Promise<{ id: string; model: WorldBody }>;
  }

  async upload(
    file: Blob,
    filename: string,
  ): Promise<{ id: string }> {
    const form = new FormData();
    form.append("file", file, filename);
    const res = await this.fetch(`${this.baseUrl}/models/save_model`, {
      method: "POST",
      body: form,
    });
    return res.json() as Promise<{ id: string }>;
  }
}
