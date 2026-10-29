import { describe, expect, it } from "vitest";
import comp from "./golden/composition.json";
import golden from "./golden/simulator.json";
import { dirichletIlrMoments, ilr, ilrInverse } from "../src/engine/composition";
import { energyProxy, printTimeProxy, trueOutcome } from "../src/engine/simulator";

type Row = (typeof golden)[number];
const MAP: [keyof Row, string][] = [
  ["strength_MPa", "strengthMPa"], ["modulus_GPa", "modulusGPa"], ["elongation_pct", "elongationPct"],
  ["p_success", "pSuccess"], ["p_clog", "pClog"], ["p_warp", "pWarp"], ["noise_cv", "noiseCv"], ["bond", "bond"],
  ["porosity", "porosity"], ["extrusion", "extrusion"], ["immiscibility", "immiscibility"],
  ["degradation", "degradation"], ["unmelted", "unmelted"], ["fiber_volume", "fiberVolume"],
  ["inclusion_volume", "inclusionVolume"],
];

function close(a: number, b: number, rel = 1e-9, abs = 1e-12): boolean {
  return Math.abs(a - b) <= abs + rel * Math.abs(b);
}

describe("simulator parity with the Python reference", () => {
  it("matches every output on 400 golden rows", () => {
    let worst = 0;
    for (const row of golden) {
      const o = trueOutcome(row.composition, row.params, row.moisture, row.batch_effect) as unknown as Record<string, number>;
      for (const [py, ts] of MAP) {
        const ref = row[py] as number;
        const err = Math.abs(o[ts] - ref) / Math.max(Math.abs(ref), 1e-12);
        worst = Math.max(worst, Math.abs(o[ts] - ref) < 1e-12 ? 0 : err);
        expect(close(o[ts], ref, 1e-9), `${py} row ${golden.indexOf(row)}: ${o[ts]} vs ${ref}`).toBe(true);
      }
      expect(close(printTimeProxy(row.params[2], row.params[3], row.params[4]), row.print_time)).toBe(true);
      expect(close(energyProxy(row.params[0], row.params[1], row.params[2], row.params[3], row.params[4]), row.energy)).toBe(true);
    }
    expect(worst).toBeLessThan(1e-9);
  });
});

describe("composition parity", () => {
  it("ilr, inverse and Dirichlet ilr moments", () => {
    comp.composition.forEach((c, i) => {
      ilr(c).forEach((v, j) => expect(close(v, comp.ilr[i][j], 1e-10, 1e-12)).toBe(true));
      ilrInverse(comp.ilr[i]).forEach((v, j) => expect(close(v, comp.ilr_inverse[i][j], 1e-10, 1e-12)).toBe(true));
    });
    const { mean, cov } = dirichletIlrMoments(comp.alpha);
    mean.forEach((v, j) => expect(close(v, comp.ilr_mean[j], 1e-9)).toBe(true));
    cov.forEach((r, p) => r.forEach((v, q) => expect(close(v, comp.ilr_cov[p][q], 1e-8, 1e-12)).toBe(true)));
  });
});
