import { ScalarFunction, MultiVariableFunction, IntegrationOptions } from '../core/types';

export class Integrator {
  static quad(
    f: ScalarFunction,
    a: number,
    b: number,
    options: IntegrationOptions = {}
  ): number {
    const { method = 'adaptive', tolerance = 1e-8, maxIterations = 100 } = options;

    switch (method) {
      case 'trapezoidal':
        return this.trapezoidal(f, a, b, maxIterations);
      case 'simpson':
        return this.simpson(f, a, b, maxIterations);
      case 'gauss':
        return this.gaussLegendre(f, a, b, options.order || 10);
      case 'adaptive':
        return this.adaptiveSimpson(f, a, b, tolerance, maxIterations);
      default:
        return this.adaptiveSimpson(f, a, b, tolerance, maxIterations);
    }
  }

  static dblquad(
    f: MultiVariableFunction,
    a: number,
    b: number,
    c: (x: number) => number,
    d: (x: number) => number,
    options: IntegrationOptions = {}
  ): number {
    const innerIntegral = (x: number): number => {
      const yLow = c(x);
      const yHigh = d(x);
      return this.quad((y: number) => f(x, y), yLow, yHigh, options);
    };

    return this.quad(innerIntegral, a, b, options);
  }

  static tplquad(
    f: MultiVariableFunction,
    a: number,
    b: number,
    c: (x: number) => number,
    d: (x: number) => number,
    e: (x: number, y: number) => number,
    g: (x: number, y: number) => number,
    options: IntegrationOptions = {}
  ): number {
    const innerIntegral2 = (x: number, y: number): number => {
      const zLow = e(x, y);
      const zHigh = g(x, y);
      return this.quad((z: number) => f(x, y, z), zLow, zHigh, options);
    };

    const innerIntegral1 = (x: number): number => {
      const yLow = c(x);
      const yHigh = d(x);
      return this.quad((y: number) => innerIntegral2(x, y), yLow, yHigh, options);
    };

    return this.quad(innerIntegral1, a, b, options);
  }

  static trapz(x: number[], y: number[]): number {
    if (x.length !== y.length) {
      throw new Error('x and y arrays must have the same length');
    }

    let integral = 0;
    for (let i = 1; i < x.length; i++) {
      const dx = x[i] - x[i - 1];
      integral += dx * (y[i] + y[i - 1]) / 2;
    }

    return integral;
  }

  private static trapezoidal(f: ScalarFunction, a: number, b: number, n: number): number {
    const h = (b - a) / n;
    let sum = (f(a) + f(b)) / 2;
    for (let i = 1; i < n; i++) {
      sum += f(a + i * h);
    }
    return sum * h;
  }

  private static simpson(f: ScalarFunction, a: number, b: number, n: number): number {
    if (n % 2 !== 0) n++;
    const h = (b - a) / n;
    let sum = f(a) + f(b);
    for (let i = 1; i < n; i++) {
      sum += f(a + i * h) * (i % 2 === 0 ? 2 : 4);
    }
    return sum * h / 3;
  }

  private static adaptiveSimpson(
    f: ScalarFunction,
    a: number,
    b: number,
    tol: number,
    maxDepth: number,
    S: number = 0,
    fa: number = 0,
    fb: number = 0,
    fc: number = 0,
    depth: number = 0
  ): number {
    if (depth === 0) {
      fa = f(a);
      fb = f(b);
      fc = f((a + b) / 2);
      S = (b - a) / 6 * (fa + 4 * fc + fb);
    }

    const c = (a + b) / 2;
    const h = b - a;
    const d = (a + c) / 2;
    const e = (c + b) / 2;
    const fd = f(d);
    const fe = f(e);

    const Sleft = h / 12 * (fa + 4 * fd + fc);
    const Sright = h / 12 * (fc + 4 * fe + fb);
    const S2 = Sleft + Sright;

    if (depth >= maxDepth || Math.abs(S2 - S) <= 15 * tol) {
      return S2 + (S2 - S) / 15;
    }

    return this.adaptiveSimpson(f, a, c, tol / 2, maxDepth, Sleft, fa, fc, fd, depth + 1) +
           this.adaptiveSimpson(f, c, b, tol / 2, maxDepth, Sright, fc, fb, fe, depth + 1);
  }

  private static gaussLegendre(f: ScalarFunction, a: number, b: number, n: number): number {
    const { roots, weights } = this.getGaussLegendrePoints(n);
    const t1 = (b - a) / 2;
    const t2 = (b + a) / 2;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += weights[i] * f(t1 * roots[i] + t2);
    }
    return t1 * sum;
  }

  private static getGaussLegendrePoints(n: number): { roots: number[]; weights: number[] } {
    const roots: number[] = [];
    const weights: number[] = [];
    const m = (n + 1) / 2;

    for (let i = 1; i <= m; i++) {
      let z = Math.cos(Math.PI * (i - 0.25) / (n + 0.5));
      let pp = 0;
      
      for (let iter = 0; iter < 10; iter++) {
        let p1 = 1;
        let p2 = 0;
        for (let j = 1; j <= n; j++) {
          const p3 = p2;
          p2 = p1;
          p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
        }
        pp = n * (z * p1 - p2) / (z * z - 1);
        const z1 = z;
        z = z1 - p1 / pp;
        if (Math.abs(z - z1) < 1e-15) break;
      }

      roots.push(z);
      weights.push(2 / ((1 - z * z) * pp * pp));
    }

    const symmetricRoots = [...roots, ...roots.slice(0, n - m).reverse().map(r => -r)];
    const symmetricWeights = [...weights, ...weights.slice(0, n - m).reverse()];

    return { roots: symmetricRoots, weights: symmetricWeights };
  }
}

export function quad(
  f: ScalarFunction,
  a: number,
  b: number,
  options?: IntegrationOptions
): number {
  return Integrator.quad(f, a, b, options);
}

export function dblquad(
  f: MultiVariableFunction,
  a: number,
  b: number,
  c: (x: number) => number,
  d: (x: number) => number,
  options?: IntegrationOptions
): number {
  return Integrator.dblquad(f, a, b, c, d, options);
}

export function tplquad(
  f: MultiVariableFunction,
  a: number,
  b: number,
  c: (x: number) => number,
  d: (x: number) => number,
  e: (x: number, y: number) => number,
  g: (x: number, y: number) => number,
  options?: IntegrationOptions
): number {
  return Integrator.tplquad(f, a, b, c, d, e, g, options);
}

export function trapz(x: number[], y: number[]): number {
  return Integrator.trapz(x, y);
}
