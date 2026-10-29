/** Print-parameter design space (port of robustprint.space). */

export interface Parameter {
  name: string;
  label: string;
  low: number;
  high: number;
  unit: string;
  default: number;
  step: number;
  hardLow: number;
  hardHigh: number;
}

export const PARAMETERS: readonly Parameter[] = [
  { name: "nozzle_temp_C", label: "Nozzle temperature", low: 210, high: 290, unit: "°C", default: 230, step: 1, hardLow: 150, hardHigh: 350 },
  { name: "bed_temp_C", label: "Bed temperature", low: 40, high: 110, unit: "°C", default: 70, step: 1, hardLow: 15, hardHigh: 150 },
  { name: "print_speed_mm_s", label: "Print speed", low: 10, high: 80, unit: "mm/s", default: 40, step: 1, hardLow: 1, hardHigh: 300 },
  { name: "layer_height_mm", label: "Layer height", low: 0.1, high: 0.35, unit: "mm", default: 0.2, step: 0.01, hardLow: 0.02, hardHigh: 1 },
  { name: "infill_pct", label: "Infill", low: 20, high: 100, unit: "%", default: 100, step: 1, hardLow: 0, hardHigh: 100 },
  { name: "raster_angle_deg", label: "Raster angle", low: 0, high: 90, unit: "°", default: 45, step: 1, hardLow: -90, hardHigh: 90 },
  { name: "flow_rate_pct", label: "Flow rate", low: 85, high: 120, unit: "%", default: 100, step: 1, hardLow: 50, hardHigh: 200 },
  { name: "cooling_fan_pct", label: "Cooling fan", low: 0, high: 100, unit: "%", default: 50, step: 1, hardLow: 0, hardHigh: 100 },
];

export const N_PARAMS = PARAMETERS.length;
export const PARAM_NAMES = PARAMETERS.map((p) => p.name);
export const LOW = PARAMETERS.map((p) => p.low);
export const HIGH = PARAMETERS.map((p) => p.high);

export function defaults(): number[] {
  return PARAMETERS.map((p) => p.default);
}

export function toUnit(x: ArrayLike<number>, low: number[] = LOW, high: number[] = HIGH): number[] {
  return Array.from(x, (v, i) => (v - low[i]) / (high[i] - low[i]));
}

export function fromUnit(u: ArrayLike<number>, low: number[] = LOW, high: number[] = HIGH): number[] {
  return Array.from(u, (v, i) => low[i] + Math.min(Math.max(v, 0), 1) * (high[i] - low[i]));
}
