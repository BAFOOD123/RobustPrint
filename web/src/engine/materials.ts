/**
 * Component properties and feedstocks (port of robustprint.physics.materials/simulator).
 * Values and their sources are documented in docs/DECISIONS.md (ADR-007 to ADR-012).
 */

export type ComponentKind = "polymer" | "fiber" | "inclusion";

export interface Component {
  name: string;
  label: string;
  description: string;
  kind: ComponentKind;
  density: number;
  modulusGPa: number;
  strengthMPa: number;
  elongationPct: number;
  tMin: number;
  tCrit: number;
  tDeg: number;
}

export const COMPONENTS: readonly Component[] = [
  { name: "pe", label: "PE", description: "Polyethylene films and bags", kind: "polymer", density: 0.95, modulusGPa: 0.8, strengthMPa: 20, elongationPct: 12, tMin: 190, tCrit: 150, tDeg: 300 },
  { name: "pet", label: "PET", description: "PET laminates and bottles", kind: "polymer", density: 1.38, modulusGPa: 2.4, strengthMPa: 55, elongationPct: 4, tMin: 255, tCrit: 230, tDeg: 295 },
  { name: "pa", label: "PA", description: "Nylon textile and film layers", kind: "polymer", density: 1.16, modulusGPa: 2.5, strengthMPa: 60, elongationPct: 15, tMin: 230, tCrit: 200, tDeg: 285 },
  { name: "fiber", label: "Fibre", description: "Short aramid-type fibres", kind: "fiber", density: 1.44, modulusGPa: 70, strengthMPa: 400, elongationPct: 3, tMin: NaN, tCrit: NaN, tDeg: 400 },
  { name: "contaminant", label: "Contam.", description: "Foam, cellulose, residues", kind: "inclusion", density: 1.2, modulusGPa: 0.05, strengthMPa: 2, elongationPct: 1, tMin: NaN, tCrit: NaN, tDeg: 235 },
];

export const N_COMPONENTS = COMPONENTS.length;
export const COMPONENT_NAMES = COMPONENTS.map((c) => c.name);

/** Pairwise incompatibility chi_ij between matrix polymers (ASSUMPTION, ADR-008). */
export const CHI: number[][] = (() => {
  const pairs: Record<string, number> = { "pe|pet": 1.0, "pe|pa": 0.9, "pet|pa": 0.5 };
  return COMPONENTS.map((a) =>
    COMPONENTS.map((b) => {
      if (a.name === b.name) return 0;
      return pairs[`${a.name}|${b.name}`] ?? pairs[`${b.name}|${a.name}`] ?? 0;
    }),
  );
})();

export interface Feedstock {
  name: string;
  mean: number[];
  concentration: number;
  moistureMedianPct: number;
  moistureLogSd: number;
}

export const ISS_NOMINAL: Feedstock = {
  name: "iss_nominal",
  mean: [0.45, 0.25, 0.15, 0.08, 0.07],
  concentration: 35,
  moistureMedianPct: 0.3,
  moistureLogSd: 0.6,
};

export const ISS_SHIFTED: Feedstock = {
  name: "iss_shifted",
  mean: [0.3, 0.3, 0.12, 0.16, 0.12],
  concentration: 35,
  moistureMedianPct: 0.3,
  moistureLogSd: 0.6,
};

export const FEEDSTOCKS: Record<string, Feedstock> = { iss_nominal: ISS_NOMINAL, iss_shifted: ISS_SHIFTED };
