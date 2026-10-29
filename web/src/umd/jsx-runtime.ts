/** Artifact build only: the automatic JSX runtime on top of the UMD React global. */
import type * as ReactNS from "react";

const R = (globalThis as unknown as { React: typeof ReactNS }).React;
type Props = Record<string, unknown> | null;
export const Fragment = R.Fragment;
export function jsx(type: Parameters<typeof R.createElement>[0], props: Props, key?: string | number) {
  return R.createElement(type, key === undefined ? props : { ...props, key });
}
export const jsxs = jsx;
