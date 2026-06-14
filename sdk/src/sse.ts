import { LagrangianError } from "./errors.js";

export async function* streamBinarySSE(
  url: string,
  init?: RequestInit,
): AsyncGenerator<string> {
  const res = await fetch(url, {
    ...init,
    headers: {
      ...(init?.headers as Record<string, string> | undefined),
      Accept: "text/event-stream",
    },
  });

  if (!res.ok || !res.body) {
    throw new LagrangianError(res.status, await res.text());
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buf += decoder.decode(value, { stream: true });
    const events = buf.split("\n\n");
    buf = events.pop() ?? "";

    for (const event of events) {
      const dataLine = event.split("\n").find((l) => l.startsWith("data:"));
      if (dataLine) {
        yield dataLine.slice(5).trim();
      }
    }
  }
}

export async function* streamSSE<T>(
  url: string,
  init?: RequestInit,
): AsyncGenerator<T> {
  const res = await fetch(url, {
    ...init,
    headers: {
      ...(init?.headers as Record<string, string> | undefined),
      Accept: "text/event-stream",
    },
  });

  if (!res.ok || !res.body) {
    throw new LagrangianError(res.status, await res.text());
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buf += decoder.decode(value, { stream: true });
    const events = buf.split("\n\n");
    buf = events.pop() ?? "";

    for (const event of events) {
      const dataLine = event
        .split("\n")
        .find((l) => l.startsWith("data:"));
      if (dataLine) {
        yield JSON.parse(dataLine.slice(5).trim()) as T;
      }
    }
  }
}
