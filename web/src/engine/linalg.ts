/** Dense linear algebra on row-major Float64Array matrices (small n, no dependencies). */

/** In-place lower Cholesky factor of an n x n SPD matrix; returns false if not PD. */
export function choleskyInPlace(A: Float64Array, n: number): boolean {
  for (let j = 0; j < n; j++) {
    let d = A[j * n + j];
    for (let k = 0; k < j; k++) d -= A[j * n + k] * A[j * n + k];
    if (!(d > 0)) return false;
    const ljj = Math.sqrt(d);
    A[j * n + j] = ljj;
    for (let i = j + 1; i < n; i++) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k++) s -= A[i * n + k] * A[j * n + k];
      A[i * n + j] = s / ljj;
    }
    for (let k = j + 1; k < n; k++) A[j * n + k] = 0;
  }
  return true;
}

/** Cholesky with increasing diagonal jitter until it succeeds. */
export function choleskyJitter(A: Float64Array, n: number): { L: Float64Array; jitter: number } {
  let jitter = 0;
  for (let attempt = 0; attempt < 8; attempt++) {
    const L = Float64Array.from(A);
    if (jitter > 0) for (let i = 0; i < n; i++) L[i * n + i] += jitter;
    if (choleskyInPlace(L, n)) return { L, jitter };
    jitter = jitter === 0 ? 1e-10 : jitter * 10;
  }
  throw new Error("Matrix is not positive definite even with jitter.");
}

/** Solve L x = b (forward substitution). */
export function solveLower(L: Float64Array, n: number, b: ArrayLike<number>): Float64Array {
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * n + k] * x[k];
    x[i] = s / L[i * n + i];
  }
  return x;
}

/** Solve L^T x = b (back substitution). */
export function solveUpperT(L: Float64Array, n: number, b: ArrayLike<number>): Float64Array {
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
    x[i] = s / L[i * n + i];
  }
  return x;
}

/** Solve (L L^T) x = b. */
export function cholSolve(L: Float64Array, n: number, b: ArrayLike<number>): Float64Array {
  return solveUpperT(L, n, solveLower(L, n, b));
}

/** Inverse of L L^T. */
export function cholInverse(L: Float64Array, n: number): Float64Array {
  const inv = new Float64Array(n * n);
  const e = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    e.fill(0);
    e[j] = 1;
    const col = cholSolve(L, n, e);
    for (let i = 0; i < n; i++) inv[i * n + j] = col[i];
  }
  return inv;
}

/** Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations. */
export function symmetricEigen(A: number[][]): { values: number[]; vectors: number[][] } {
  const n = A.length;
  const a = A.map((r) => r.slice());
  const v: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
  }
  return { values: a.map((r, i) => r[i]), vectors: v };
}
