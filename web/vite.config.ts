import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";
import { viteSingleFile } from "vite-plugin-singlefile";

const umd = (p: string) => fileURLToPath(new URL(`./src/umd/${p}`, import.meta.url));

// Single self-contained index.html: the Virtual Lab runs entirely in the browser,
// with the heavy computations in an inlined Web Worker. `--mode artifact` builds the
// claude.ai Artifact variant, which takes React from cdnjs script tags instead of the bundle.
export default defineConfig(({ mode }) => ({
  plugins: [react(), viteSingleFile()],
  base: "./",
  resolve:
    mode === "artifact"
      ? { alias: [{ find: /^react$/, replacement: umd("react.ts") }, { find: /^react\/jsx-runtime$/, replacement: umd("jsx-runtime.ts") }, { find: /^react-dom\/client$/, replacement: umd("react-dom-client.ts") }] }
      : {},
  build: { target: "es2022", assetsInlineLimit: 100_000_000, chunkSizeWarningLimit: 5000, outDir: mode === "artifact" ? "dist-artifact" : "dist" },
  // Classic (IIFE) worker: module workers from data: URLs are refused when the file is opened from disk.
  worker: { format: "iife" },
  test: { include: ["tests/**/*.test.ts"], environment: "node", testTimeout: 60_000 },
}));
