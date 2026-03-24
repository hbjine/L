import { Vector, ScalarFunction, MultiVariableFunction, RootFindingOptions } from '../core/types';
import { MatrixOps } from '../core/Matrix';
import { LinearSystemSolver } from '../linalg/LinearSystemSolver';

export class NonLinearSolver {
  static root(
    f: ScalarFunction,
    a: number,
    b: number,
    options: RootFindingOptions = {}
  ): number {
    const { method = 'brentq', tolerance = 1e-8, maxIterations = 100 } = options;

    switch (method) {
      case 'bisection':
        return this.bisection(f, a, b, tolerance, maxIterations);
      case 'newton':
        return this.newtonRaphson(f, (a + b) / 2, tolerance, maxIterations);
      case 'secant':
        return this.secant(f, a, b, tolerance, maxIterations);
      case 'brentq':
        return this.brentq(f, a, b, tolerance, maxIterations);
      default:
        return this.brentq(f, a, b, tolerance, maxIterations);
    }
  }

  static fsolve(
    f: (x: Vector) => Vector,
    x0: Vector,
    options: RootFindingOptions = {}
  ): Vector {
    const { tolerance = 1e-8, maxIterations = 100 } = options;
    let x = [...x0];
    const n = x.length;

    for (let iter = 0; iter < maxIterations; iter++) {
      const fx = f(x);
      const error = Math.max(...fx.map(Math.abs));
      
      if (error < tolerance) {
        return x;
      }

      const J = this.approximateJacobian(f, x);
      const dx = LinearSystemSolver.solve(J, fx.map(val => -val));
      x = x.map((val, i) => val + dx[i]);
    }

    return x;
  }

  static fixedPoint(
    g: (x: number) => number,
    x0: number,
    tolerance: number = 1e-8,
    maxIterations: number = 100
  ): number {
    let x = x0;
    for (let iter = 0; iter < maxIterations; iter++) {
      const xNew = g(x);
      if (Math.abs(xNew - x) < tolerance) {
        return xNew;
      }
      x = xNew;
    }
    return x;
  }

  private static bisection(
    f: ScalarFunction,
    a: number,
    b: number,
    tol: number,
    maxIter: number
  ): number {
    let fa = f(a);
    let fb = f(b);

    if (fa * fb >= 0) {
      throw new Error('Function has same sign at both endpoints');
    }

    for (let iter = 0; iter < maxIter; iter++) {
      const c = (a + b) / 2;
      const fc = f(c);

      if (Math.abs(fc) < tol || (b - a) / 2 < tol) {
        return c;
      }

      if (fa * fc < 0) {
        b = c;
        fb = fc;
      } else {
        a = c;
        fa = fc;
      }
    }

    return (a + b) / 2;
  }

  private static newtonRaphson(
    f: ScalarFunction,
    x0: number,
    tol: number,
    maxIter: number
  ): number {
    let x = x0;
    for (let iter = 0; iter < maxIter; iter++) {
      const fx = f(x);
      if (Math.abs(fx) < tol) {
        return x;
      }
      const df = this.derivative(f, x);
      x -= fx / df;
    }
    return x;
  }

  private static secant(
    f: ScalarFunction,
    a: number,
    b: number,
    tol: number,
    maxIter: number
  ): number {
    let x0 = a;
    let x1 = b;
    let f0 = f(x0);
    let f1 = f(x1);

    for (let iter = 0; iter < maxIter; iter++) {
      if (Math.abs(f1 - f0) < 1e-15) {
        break;
      }
      const x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
      if (Math.abs(x2 - x1) < tol) {
        return x2;
      }
      x0 = x1;
      x1 = x2;
      f0 = f1;
      f1 = f(x1);
    }
    return x1;
  }

  private static brentq(
    f: ScalarFunction,
    a: number,
    b: number,
    tol: number,
    maxIter: number
  ): number {
    let fa = f(a);
    let fb = f(b);

    if (fa * fb >= 0) {
      throw new Error('Function has same sign at both endpoints');
    }

    if (Math.abs(fa) < Math.abs(fb)) {
      [a, b] = [b, a];
      [fa, fb] = [fb, fa];
    }

    let c = a;
    let fc = fa;
    let s = b;
    let d = 0;
    let mflag = true;

    for (let iter = 0; iter < maxIter; iter++) {
      if (Math.abs(b - a) < tol) {
        return b;
      }

      if (fa !== fc && fb !== fc) {
        const s1 = (a * fb * fc) / ((fa - fb) * (fa - fc));
        const s2 = (b * fa * fc) / ((fb - fa) * (fb - fc));
        const s3 = (c * fa * fb) / ((fc - fa) * (fc - fb));
        s = s1 + s2 + s3;
      } else {
        s = b - fb * (b - a) / (fb - fa);
      }

      const cond1 = (s - (3 * a + b) / 4) * (s - b) >= 0;
      const cond2 = mflag && Math.abs(s - b) >= Math.abs(b - c) / 2;
      const cond3 = !mflag && Math.abs(s - b) >= Math.abs(c - d) / 2;
      const cond4 = mflag && Math.abs(b - c) < tol;
      const cond5 = !mflag && Math.abs(c - d) < tol;

      if (cond1 || cond2 || cond3 || cond4 || cond5) {
        s = (a + b) / 2;
        mflag = true;
      } else {
        mflag = false;
      }

      const fs = f(s);
      d = c;
      c = b;
      fc = fb;

      if (fa * fs < 0) {
        b = s;
        fb = fs;
      } else {
        a = s;
        fa = fs;
      }

      if (Math.abs(fa) < Math.abs(fb)) {
        [a, b] = [b, a];
        [fa, fb] = [fb, fa];
      }
    }

    return b;
  }

  private static derivative(f: ScalarFunction, x: number, eps: number = 1e-8): number {
    return (f(x + eps) - f(x - eps)) / (2 * eps);
  }

  private static approximateJacobian(
    f: (x: Vector) => Vector,
    x: Vector,
    eps: number = 1e-8
  ): number[][] {
    const n = x.length;
    const J = MatrixOps.zeros(n, n);
    const fx = f(x);

    for (let j = 0; j < n; j++) {
      const xEps = [...x];
      xEps[j] += eps;
      const fxEps = f(xEps);
      for (let i = 0; i < n; i++) {
        J[i][j] = (fxEps[i] - fx[i]) / eps;
      }
    }

    return J;
  }
}

export function root(
  fun: ScalarFunction,
  a: number,
  b: number,
  options?: RootFindingOptions
): number {
  return NonLinearSolver.root(fun, a, b, options);
}

export function fsolve(
  fun: (x: Vector) => Vector,
  x0: Vector,
  options?: RootFindingOptions
): Vector {
  return NonLinearSolver.fsolve(fun, x0, options);
}

export function brentq(
  f: ScalarFunction,
  a: number,
  b: number,
  options?: RootFindingOptions
): number {
  return NonLinearSolver.root(f, a, b, { ...options, method: 'brentq' });
}
