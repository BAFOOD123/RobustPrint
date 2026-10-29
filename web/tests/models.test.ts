import { describe, expect, it } from "vitest";
import { generateCampaign } from "../src/engine/campaign";
import { ISS_NOMINAL } from "../src/engine/materials";

describe("virtual campaign", () => {
  it("job means are unbiased measurements of the true strength", () => {
    const jobs = generateCampaign(ISS_NOMINAL, 150, 4, 11).filter((j) => j.success);
    const r = jobs.map((j) => Math.log(j.strengthMPa / (j.trueStrengthMPa as number)));
    const m = r.reduce((a, b) => a + b, 0) / r.length;
    const sd = Math.sqrt(r.reduce((s, v) => s + (v - m) ** 2, 0) / r.length);
    expect(Math.abs(m)).toBeLessThan(0.03);
    expect(sd).toBeLessThan(0.15);
    expect(jobs.every((j) => Number.isFinite(j.strengthSdMPa))).toBe(true);
  });
});
