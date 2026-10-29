/** Artifact build only: React from the UMD script tag (window.React) instead of the bundle. */
import type * as ReactNS from "react";

const R = (globalThis as unknown as { React: typeof ReactNS }).React;
export default R;
export const { StrictMode, Fragment, createContext, createElement, useContext, useEffect, useLayoutEffect, useMemo, useReducer, useRef, useState } = R;
