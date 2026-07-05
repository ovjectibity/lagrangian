import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  sourcemap: true,
  // protobufjs has no "exports" map, so its extensionless subpath imports
  // (e.g. "protobufjs/minimal") fail under strict Node ESM resolution if left
  // external. Bundle it into dist so esbuild resolves it at build time.
  noExternal: ["protobufjs"],
});
