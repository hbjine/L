import { Vector, InterpolationOptions } from '../core/types';
import { MatrixOps } from '../core/Matrix';
import { LinearSystemSolver } from '../linalg/LinearSystemSolver';

export class Interpolator {
  private x: Vector;
  private y: Vector;
  private coefficients: Vector | null = null;
  private method: string;
  private order: number;

  constructor(x: Vector, y: Vector, options: InterpolationOptions = {}) {
    if (x.length !== y.length) {
      throw new Error('x and y arrays must have the same length');
    }
    if (x.length < 2) {
      throw new Error('At least two data points are required');
    }

    this.x = [...x];
    this.y = [...y];
    this.method = options.method || 'linear';
    this.order = options.order || 3;

    this.sortData();
    this.computeCoefficients();
  }

  private sortData(): void {
    const indices = this.x
      .map((val, idx) => ({ val, idx }))
      .sort((a, b) => a.val - b.val)
      .map(item => item.idx);

    this.x = indices.map(i => this.x[i]);
    this.y = indices.map(i => this.y[i]);
  }

  private computeCoefficients(): void {
    const n = this.x.length;

    switch (this.method) {
      case 'linear':
        this.coefficients = null;
        break;
      case 'nearest':
        this.coefficients = null;
        break;
      case 'spline':
        this.coefficients = this.computeSplineCoefficients();
        break;
      case 'polynomial':
        this.coefficients = this.computePolynomialCoefficients();
        break;
      default:
        this.coefficients = null;
    }
  }

  private computeSplineCoefficients(): Vector {
    const n = this.x.length;
    const h = new Array(n - 1).fill(0);
    const alpha = new Array(n - 1).fill(0);
    const l = new Array(n).fill(1);
    const mu = new Array(n).fill(0);
    const z = new Array(n).fill(0);
    const c = new Array(n).fill(0);
    const b = new Array(n - 1).fill(0);
    const d = new Array(n - 1).fill(0);

    for (let i = 0; i < n - 1; i++) {
      h[i] = this.x[i + 1] - this.x[i];
    }

    for (let i = 1; i < n - 1; i++) {
      alpha[i] = 3 * (this.y[i + 1] - this.y[i]) / h[i] - 3 * (this.y[i] - this.y[i - 1]) / h[i - 1];
    }

    for (let i = 1; i < n - 1; i++) {
      l[i] = 2 * (this.x[i + 1] - this.x[i - 1]) - h[i - 1] * mu[i - 1];
      mu[i] = h[i] / l[i];
      z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    for (let j = n - 2; j >= 0; j--) {
      c[j] = z[j] - mu[j] * c[j + 1];
      b[j] = (this.y[j + 1] - this.y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
      d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
    }

    const coefficients: number[] = [];
    for (let i = 0; i < n - 1; i++) {
      coefficients.push(this.y[i], b[i], c[i], d[i]);
    }

    return coefficients;
  }

  private computePolynomialCoefficients(): Vector {
    const n = this.x.length;
    const order = Math.min(this.order, n - 1);
    const A = MatrixOps.zeros(n, order + 1);

    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= order; j++) {
        A[i][j] = Math.pow(this.x[i], j);
      }
    }

    return LinearSystemSolver.leastSquares(A, this.y);
  }

  interpolate(x: number): number {
    const n = this.x.length;

    if (x <= this.x[0]) {
      return this.extrapolateLeft(x);
    }
    if (x >= this.x[n - 1]) {
      return this.extrapolateRight(x);
    }

    const i = this.findInterval(x);

    switch (this.method) {
      case 'linear':
        return this.linearInterpolation(x, i);
      case 'nearest':
        return this.nearestInterpolation(x, i);
      case 'spline':
        return this.splineInterpolation(x, i);
      case 'polynomial':
        return this.polynomialInterpolation(x);
      default:
        return this.linearInterpolation(x, i);
    }
  }

  interpolateMany(xValues: Vector): Vector {
    return xValues.map(x => this.interpolate(x));
  }

  private findInterval(x: number): number {
    let low = 0;
    let high = this.x.length - 1;

    while (high - low > 1) {
      const mid = Math.floor((low + high) / 2);
      if (this.x[mid] <= x) {
        low = mid;
      } else {
        high = mid;
      }
    }

    return low;
  }

  private linearInterpolation(x: number, i: number): number {
    const t = (x - this.x[i]) / (this.x[i + 1] - this.x[i]);
    return this.y[i] + t * (this.y[i + 1] - this.y[i]);
  }

  private nearestInterpolation(x: number, i: number): number {
    const d1 = x - this.x[i];
    const d2 = this.x[i + 1] - x;
    return d1 <= d2 ? this.y[i] : this.y[i + 1];
  }

  private splineInterpolation(x: number, i: number): number {
    if (!this.coefficients) {
      return this.linearInterpolation(x, i);
    }

    const t = x - this.x[i];
    const idx = i * 4;
    return this.coefficients[idx] +
           this.coefficients[idx + 1] * t +
           this.coefficients[idx + 2] * t * t +
           this.coefficients[idx + 3] * t * t * t;
  }

  private polynomialInterpolation(x: number): number {
    if (!this.coefficients) {
      return 0;
    }

    let result = 0;
    for (let i = this.coefficients.length - 1; i >= 0; i--) {
      result = result * x + this.coefficients[i];
    }
    return result;
  }

  private extrapolateLeft(x: number): number {
    if (this.method === 'polynomial' && this.coefficients) {
      return this.polynomialInterpolation(x);
    }
    const slope = (this.y[1] - this.y[0]) / (this.x[1] - this.x[0]);
    return this.y[0] + slope * (x - this.x[0]);
  }

  private extrapolateRight(x: number): number {
    if (this.method === 'polynomial' && this.coefficients) {
      return this.polynomialInterpolation(x);
    }
    const n = this.x.length;
    const slope = (this.y[n - 1] - this.y[n - 2]) / (this.x[n - 1] - this.x[n - 2]);
    return this.y[n - 1] + slope * (x - this.x[n - 1]);
  }

  static griddata(
    x: Vector,
    y: Vector,
    z: Vector,
    xi: Vector,
    yi: Vector,
    method: 'linear' | 'nearest' = 'linear'
  ): Vector {
    if (x.length !== y.length || x.length !== z.length) {
      throw new Error('x, y, and z arrays must have the same length');
    }

    const result: number[] = [];

    for (let k = 0; k < xi.length; k++) {
      const xi_k = xi[k];
      const yi_k = yi[k];

      let minDist = Infinity;
      let nearestZ = 0;
      let nearestIdx = 0;

      for (let i = 0; i < x.length; i++) {
        const dist = Math.sqrt((x[i] - xi_k) ** 2 + (y[i] - yi_k) ** 2);
        if (dist < minDist) {
          minDist = dist;
          nearestZ = z[i];
          nearestIdx = i;
        }
      }

      if (method === 'nearest') {
        result.push(nearestZ);
      } else {
        const neighbors: number[] = [nearestIdx];
        const dists: number[] = [minDist];

        for (let i = 0; i < x.length; i++) {
          if (i === nearestIdx) continue;
          const dist = Math.sqrt((x[i] - xi_k) ** 2 + (y[i] - yi_k) ** 2);
          if (neighbors.length < 3) {
            neighbors.push(i);
            dists.push(dist);
          } else {
            const maxDist = Math.max(...dists);
            if (dist < maxDist) {
              const maxIdx = dists.indexOf(maxDist);
              neighbors[maxIdx] = i;
              dists[maxIdx] = dist;
            }
          }
        }

        const x1 = x[neighbors[0]], y1 = y[neighbors[0]], z1 = z[neighbors[0]];
        const x2 = x[neighbors[1]], y2 = y[neighbors[1]], z2 = z[neighbors[1]];
        const x3 = x[neighbors[2]], y3 = y[neighbors[2]], z3 = z[neighbors[2]];

        const A = [
          [x1, y1, 1],
          [x2, y2, 1],
          [x3, y3, 1]
        ];
        const planeCoeffs = LinearSystemSolver.solve(A, [z1, z2, z3]);
        const interpolatedZ = planeCoeffs[0] * xi_k + planeCoeffs[1] * yi_k + planeCoeffs[2];
        result.push(interpolatedZ);
      }
    }

    return result;
  }
}

export function interp1d(
  x: Vector,
  y: Vector,
  options?: InterpolationOptions
): (x: number | Vector) => number | Vector {
  const interpolator = new Interpolator(x, y, options);
  return (xi: number | Vector): number | Vector => {
    if (typeof xi === 'number') {
      return interpolator.interpolate(xi);
    } else {
      return interpolator.interpolateMany(xi);
    }
  };
}

export function splrep(x: Vector, y: Vector): { tck: Vector[] } {
  const interpolator = new Interpolator(x, y, { method: 'spline' });
  const n = x.length;
  const t = [...x];
  const c = interpolator['coefficients'] || [];
  const k = [3];
  return { tck: [t, c, k] };
}

export function splev(x: number | Vector, tck: { tck: Vector[] }): number | Vector {
  const [t, c, k] = tck.tck;
  const interpolator = new Interpolator(t.slice(0, t.length - 3), t.slice(0, t.length - 3), { method: 'spline' });
  
  if (typeof x === 'number') {
    return interpolator.interpolate(x);
  } else {
    return x.map(xi => interpolator.interpolate(xi));
  }
}

export function griddata(
  x: Vector,
  y: Vector,
  z: Vector,
  xi: Vector,
  yi: Vector,
  method: 'linear' | 'nearest' = 'linear'
): Vector {
  return Interpolator.griddata(x, y, z, xi, yi, method);
}
