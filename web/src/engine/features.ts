/** Feature construction and the simplified physics prior (ports of models.features / baselines). */

import { closure, ilr } from "./composition";
import { COMPONENTS } from "./materials";
import { toUnit } from "./space";
import { sectionFraction, tsaiHillFactor } from "./simulator";

/** ilr(c) (4) + settings scaled to the search box (8) + log moisture (1). */
export function features(composition: ArrayLike<number>, params: ArrayLike<number>, moisture: number): number[] {
  return [...ilr(closure(composition)), ...toUnit(params), Math.log(Math.max(moisture, 1e-3))];
}

/** Mass-fraction rule of mixtures of bulk strengths. */
export function ruleOfMixturesStrength(composition: ArrayLike<number>): number {
  const c = closure(composition);
  return COMPONENTS.reduce((s, comp, i) => s + c[i] * comp.strengthMPa, 0);
}

/**
 * Deliberately incomplete physics prior on log strength: rule of mixtures x infill
 * section x Tsai-Hill raster factor (t = q = 0.5). No blend, thermal or failure physics.
 */
export function physicsPrior(composition: ArrayLike<number>, params: ArrayLike<number>): number {
  return Math.log(ruleOfMixturesStrength(composition)) + Math.log(sectionFraction(params[4])) + Math.log(tsaiHillFactor(params[5], 0.5, 0.5));
}
