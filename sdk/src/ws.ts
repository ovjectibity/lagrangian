import { LagrangianError } from "./errors.js";

/** Convert an http(s) URL to a ws(s) URL. */
export function httpToWs(url: string): string {
  return url.replace(/^http/, "ws");
}

/**
 * Open a WebSocket and expose its incoming messages as an async generator.
 * Yields each message's `data` (a string for text frames, an ArrayBuffer for
 * binary frames). Ends when the socket closes; throws on a socket error.
 *
 * Uses the global `WebSocket` (browsers and Node >= 22).
 */
async function* streamWSRaw(
  url: string,
  binary: boolean,
): AsyncGenerator<string | ArrayBuffer> {
  const ws = new WebSocket(url);
  if (binary) ws.binaryType = "arraybuffer";

  const queue: (string | ArrayBuffer)[] = [];
  let closed = false;
  let failure: LagrangianError | null = null;
  let wake: (() => void) | null = null;
  const notify = () => {
    if (wake) {
      wake();
      wake = null;
    }
  };

  ws.onmessage = (ev: MessageEvent) => {
    queue.push(ev.data as string | ArrayBuffer);
    notify();
  };
  ws.onclose = () => {
    closed = true;
    notify();
  };
  ws.onerror = () => {
    failure = new LagrangianError(0, "WebSocket error");
    closed = true;
    notify();
  };

  try {
    while (true) {
      if (queue.length > 0) {
        yield queue.shift()!;
        continue;
      }
      if (closed) break;
      await new Promise<void>((resolve) => {
        wake = resolve;
      });
    }
    if (failure) throw failure;
  } finally {
    try {
      ws.close();
    } catch {
      // ignore
    }
  }
}

/** Stream binary frames from a WebSocket endpoint (http(s) URL accepted). */
export async function* streamBinaryWS(
  url: string,
): AsyncGenerator<ArrayBuffer> {
  for await (const msg of streamWSRaw(httpToWs(url), true)) {
    if (msg instanceof ArrayBuffer) yield msg;
  }
}

/** Stream JSON text frames from a WebSocket endpoint (http(s) URL accepted). */
export async function* streamWS<T>(url: string): AsyncGenerator<T> {
  for await (const msg of streamWSRaw(httpToWs(url), false)) {
    if (typeof msg === "string") yield JSON.parse(msg) as T;
  }
}
