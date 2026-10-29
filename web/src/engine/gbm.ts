/**
 * Print-success classifier: gradient-boosted trees with isotonic recalibration, a port of
 * SuccessClassifier("gbm_isotonic") = CalibratedClassifierCV(HistGradientBoostingClassifier(
 * max_iter=150, learning_rate=0.05, max_leaf_nodes=15), method="isotonic", cv=3).
 * Trees use exact thresholds (equivalent to scikit-learn's binning below 256 distinct values),
 * best-first growth, min_samples_leaf=20 and Newton leaf values. Folds are stratified but not
 * identical to scikit-learn's, so predictions agree statistically, not bit for bit.
 */

const MAX_ITER = 150;
const LEARNING_RATE = 0.05;
const MAX_LEAF_NODES = 15;
const MIN_SAMPLES_LEAF = 20;
const MIN_HESSIAN = 1e-3;
const N_FOLDS = 3;

/** A leaf has feature = -1; a split node sends x[feature] <= threshold to the left. */
interface Node { feature: number; threshold: number; value: number; left: Node | null; right: Node | null }

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

function predictTree(node: Node, x: ArrayLike<number>): number {
  while (node.feature >= 0) node = x[node.feature] <= node.threshold ? node.left! : node.right!;
  return node.value;
}

interface Split { gain: number; feature: number; threshold: number; left: number[]; right: number[] }

/** Training data shared by all trees: features, per-feature row order (sorted once) and a scratch mask. */
interface Data { X: number[][]; sorted: Int32Array[]; mask: Uint8Array }

function bestSplit(data: Data, g: Float64Array, h: Float64Array, idx: number[]): Split | null {
  const n = idx.length;
  if (n < 2 * MIN_SAMPLES_LEAF) return null;
  const { X, sorted, mask } = data;
  let G = 0;
  let H = 0;
  for (const i of idx) {
    G += g[i];
    H += h[i];
    mask[i] = 1;
  }
  const parent = (G * G) / H;
  let best: Split | null = null;
  for (let f = 0; f < sorted.length; f++) {
    let gl = 0;
    let hl = 0;
    let k = 0;
    let prev = -1;
    for (const i of sorted[f]) {
      if (!mask[i]) continue;
      if (prev >= 0) {
        // Candidate split between the previous member row and this one.
        const xv = X[prev][f];
        const xn = X[i][f];
        const hr = H - hl;
        if (xv !== xn && k >= MIN_SAMPLES_LEAF && n - k >= MIN_SAMPLES_LEAF && hl >= MIN_HESSIAN && hr >= MIN_HESSIAN) {
          const gr = G - gl;
          const gain = (gl * gl) / hl + (gr * gr) / hr - parent;
          if (gain > 1e-12 && (!best || gain > best.gain)) best = { gain, feature: f, threshold: (xv + xn) / 2, left: [], right: [] };
        }
      }
      gl += g[i];
      hl += h[i];
      k += 1;
      prev = i;
    }
  }
  for (const i of idx) mask[i] = 0;
  if (best) for (const i of idx) (X[i][best.feature] <= best.threshold ? best.left : best.right).push(i);
  return best;
}

/** Grow one regression tree on (g, h) best-first (largest gain first), up to MAX_LEAF_NODES leaves. */
function growTree(data: Data, g: Float64Array, h: Float64Array, idx: number[]): Node {
  const leaf = (rows: number[]): Node => {
    let G = 0;
    let H = 0;
    for (const i of rows) {
      G += g[i];
      H += h[i];
    }
    return { feature: -1, threshold: 0, value: (-LEARNING_RATE * G) / Math.max(H, 1e-12), left: null, right: null };
  };
  const root = leaf(idx);
  const open: { node: Node; split: Split }[] = [];
  const consider = (node: Node, rows: number[]) => {
    const split = bestSplit(data, g, h, rows);
    if (split) open.push({ node, split });
  };
  consider(root, idx);
  for (let leaves = 1; leaves < MAX_LEAF_NODES && open.length > 0; leaves++) {
    let pick = 0;
    for (let k = 1; k < open.length; k++) if (open[k].split.gain > open[pick].split.gain) pick = k;
    const { node, split } = open.splice(pick, 1)[0];
    node.feature = split.feature;
    node.threshold = split.threshold;
    node.left = leaf(split.left);
    node.right = leaf(split.right);
    consider(node.left, split.left);
    consider(node.right, split.right);
  }
  return root;
}

/** Binary log-loss gradient boosting (exported for the parity test against scikit-learn). */
export class Booster {
  private base = 0;
  private trees: Node[] = [];

  fit(X: number[][], y: number[]): this {
    const n = y.length;
    const rate = Math.min(Math.max(y.reduce((a, b) => a + b, 0) / n, 1e-6), 1 - 1e-6);
    this.base = Math.log(rate / (1 - rate));
    const raw = new Float64Array(n).fill(this.base);
    const g = new Float64Array(n);
    const h = new Float64Array(n);
    const all = Array.from({ length: n }, (_, i) => i);
    const data: Data = {
      X,
      sorted: X[0].map((_, f) => Int32Array.from(all.slice().sort((a, b) => X[a][f] - X[b][f]))),
      mask: new Uint8Array(n),
    };
    for (let it = 0; it < MAX_ITER; it++) {
      for (let i = 0; i < n; i++) {
        const p = sigmoid(raw[i]);
        g[i] = p - y[i];
        h[i] = p * (1 - p);
      }
      const tree = growTree(data, g, h, all);
      this.trees.push(tree);
      for (let i = 0; i < n; i++) raw[i] += predictTree(tree, X[i]);
    }
    return this;
  }

  decision(x: ArrayLike<number>): number {
    let z = this.base;
    for (const t of this.trees) z += predictTree(t, x);
    return z;
  }
}

/** Increasing isotonic regression (pool adjacent violators), predicted by linear interpolation and clipped. */
export class Isotonic {
  xs: number[] = [];
  ys: number[] = [];

  fit(x: number[], y: number[]): this {
    const order = x.map((_, i) => i).sort((a, b) => x[a] - x[b]);
    // Ties in x are averaged first, as in scikit-learn.
    const ux: number[] = [];
    const uy: number[] = [];
    const uw: number[] = [];
    for (const i of order) {
      if (ux.length && x[i] === ux[ux.length - 1]) {
        const k = ux.length - 1;
        uy[k] = (uy[k] * uw[k] + y[i]) / (uw[k] + 1);
        uw[k] += 1;
      } else {
        ux.push(x[i]);
        uy.push(y[i]);
        uw.push(1);
      }
    }
    const val: number[] = [];
    const wt: number[] = [];
    const size: number[] = [];
    for (let i = 0; i < ux.length; i++) {
      val.push(uy[i]);
      wt.push(uw[i]);
      size.push(1);
      while (val.length > 1 && val[val.length - 2] > val[val.length - 1]) {
        const w = wt[wt.length - 2] + wt[wt.length - 1];
        const v = (val[val.length - 2] * wt[wt.length - 2] + val[val.length - 1] * wt[wt.length - 1]) / w;
        const s = size[size.length - 2] + size[size.length - 1];
        val.splice(-2, 2, v);
        wt.splice(-2, 2, w);
        size.splice(-2, 2, s);
      }
    }
    this.xs = ux;
    this.ys = [];
    val.forEach((v, k) => {
      for (let j = 0; j < size[k]; j++) this.ys.push(v);
    });
    return this;
  }

  predict(x: number): number {
    const { xs, ys } = this;
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
    let lo = 0;
    let hi = xs.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xs[mid] <= x) lo = mid;
      else hi = mid;
    }
    const t = (x - xs[lo]) / (xs[hi] - xs[lo]);
    return ys[lo] + t * (ys[hi] - ys[lo]);
  }
}

export class GbmSuccessClassifier {
  constant: number | null = null;
  private members: { booster: Booster; calibrator: Isotonic }[] = [];

  fit(X: number[][], success: boolean[]): this {
    const n = X.length;
    const y: number[] = success.map((s) => (s ? 1 : 0));
    const pos = y.reduce((a, b) => a + b, 0);
    this.members = [];
    if (pos === 0 || pos === n) {
      this.constant = (pos + 1) / (n + 2);
      return this;
    }
    this.constant = null;
    // Stratified folds: each class is dealt to the folds in order.
    const fold = new Array<number>(n);
    let kPos = 0;
    let kNeg = 0;
    for (let i = 0; i < n; i++) fold[i] = y[i] ? kPos++ % N_FOLDS : kNeg++ % N_FOLDS;
    for (let f = 0; f < N_FOLDS; f++) {
      const tr = [...Array(n).keys()].filter((i) => fold[i] !== f);
      const te = [...Array(n).keys()].filter((i) => fold[i] === f);
      const trY = tr.map((i) => y[i]);
      if (trY.every((v) => v === trY[0])) continue;
      const booster = new Booster().fit(tr.map((i) => X[i]), trY);
      const calibrator = new Isotonic().fit(te.map((i) => booster.decision(X[i])), te.map((i) => y[i]));
      this.members.push({ booster, calibrator });
    }
    if (this.members.length === 0) this.constant = (pos + 1) / (n + 2);
    return this;
  }

  predictProba(X: number[][]): Float64Array {
    const out = new Float64Array(X.length);
    if (this.constant !== null) return out.fill(this.constant);
    for (let i = 0; i < X.length; i++) {
      let s = 0;
      for (const m of this.members) s += Math.min(Math.max(m.calibrator.predict(m.booster.decision(X[i])), 0), 1);
      out[i] = s / this.members.length;
    }
    return out;
  }
}
