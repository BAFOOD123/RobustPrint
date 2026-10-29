/** Artifact build only: createRoot from the UMD script tag (window.ReactDOM). */
import type * as ClientNS from "react-dom/client";

export const { createRoot } = (globalThis as unknown as { ReactDOM: typeof ClientNS }).ReactDOM;
