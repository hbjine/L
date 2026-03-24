import { Vector, VectorFunction, ODESolution, ODEOptions } from '../core/types';

export class ODESolver {
  static solveIVP(
    f: VectorFunction,
    tSpan: [number, number],
    y0: Vector,
    options: ODEOptions = {}
  ): ODESolution {
    const {
      method = 'rk45',
      stepSize = 0.1,
      tolerance = 1e-6,
      maxSteps = 10000
    } = options;

    switch (method) {
      case 'euler':
        return this.euler(f, tSpan, y0, stepSize, maxSteps);
      case 'rk4':
        return this.rk4(f, tSpan, y0, stepSize, maxSteps);
      case 'rk45':
        return this.rk45(f, tSpan, y0, tolerance, maxSteps);
      case 'bdf':
        return this.bdf(f, tSpan, y0, stepSize, maxSteps);
      default:
        return this.rk45(f, tSpan, y0, tolerance, maxSteps);
    }
  }

  private static euler(
    f: VectorFunction,
    tSpan: [number, number],
    y0: Vector,
    h: number,
    maxSteps: number
  ): ODESolution {
    const t: number[] = [tSpan[0]];
    const y: number[][] = [[...y0]];
    let currentT = tSpan[0];
    let currentY = [...y0];
    let steps = 0;

    while (currentT < tSpan[1] && steps < maxSteps) {
      const dt = Math.min(h, tSpan[1] - currentT);
      const dy = f(currentT, currentY);
      currentY = currentY.map((val, i) => val + dt * dy[i]);
      currentT += dt;
      t.push(currentT);
      y.push([...currentY]);
      steps++;
    }

    return { t, y };
  }

  private static rk4(
    f: VectorFunction,
    tSpan: [number, number],
    y0: Vector,
    h: number,
    maxSteps: number
  ): ODESolution {
    const t: number[] = [tSpan[0]];
    const y: number[][] = [[...y0]];
    let currentT = tSpan[0];
    let currentY = [...y0];
    let steps = 0;

    while (currentT < tSpan[1] && steps < maxSteps) {
      const dt = Math.min(h, tSpan[1] - currentT);
      
      const k1 = f(currentT, currentY);
      const k2 = f(currentT + dt / 2, currentY.map((val, i) => val + dt / 2 * k1[i]));
      const k3 = f(currentT + dt / 2, currentY.map((val, i) => val + dt / 2 * k2[i]));
      const k4 = f(currentT + dt, currentY.map((val, i) => val + dt * k3[i]));
      
      currentY = currentY.map((val, i) => 
        val + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
      );
      
      currentT += dt;
      t.push(currentT);
      y.push([...currentY]);
      steps++;
    }

    return { t, y };
  }

  private static rk45(
    f: VectorFunction,
    tSpan: [number, number],
    y0: Vector,
    tol: number,
    maxSteps: number
  ): ODESolution {
    const t: number[] = [tSpan[0]];
    const y: number[][] = [[...y0]];
    let currentT = tSpan[0];
    let currentY = [...y0];
    let h = 1e-6;
    let steps = 0;

    const a2 = 1 / 4, a3 = 3 / 8, a4 = 12 / 13, a5 = 1, a6 = 1 / 2;
    const b21 = 1 / 4;
    const b31 = 3 / 32, b32 = 9 / 32;
    const b41 = 1932 / 2197, b42 = -7200 / 2197, b43 = 7296 / 2197;
    const b51 = 439 / 216, b52 = -8, b53 = 3680 / 513, b54 = -845 / 4104;
    const b61 = -8 / 27, b62 = 2, b63 = -3544 / 2565, b64 = 1859 / 4104, b65 = -11 / 40;
    const c1 = 16 / 135, c3 = 6656 / 12825, c4 = 28561 / 56430, c5 = -9 / 50, c6 = 2 / 55;
    const dc1 = c1 - 25 / 216, dc3 = c3 - 1408 / 2565, dc4 = c4 - 2197 / 4104, dc5 = c5 + 1 / 5;

    while (currentT < tSpan[1] && steps < maxSteps) {
      const dt = Math.min(h, tSpan[1] - currentT);
      
      const k1 = f(currentT, currentY);
      const k2 = f(currentT + a2 * dt, currentY.map((val, i) => val + b21 * dt * k1[i]));
      const k3 = f(currentT + a3 * dt, currentY.map((val, i) => val + (b31 * k1[i] + b32 * k2[i]) * dt));
      const k4 = f(currentT + a4 * dt, currentY.map((val, i) => val + (b41 * k1[i] + b42 * k2[i] + b43 * k3[i]) * dt));
      const k5 = f(currentT + a5 * dt, currentY.map((val, i) => val + (b51 * k1[i] + b52 * k2[i] + b53 * k3[i] + b54 * k4[i]) * dt));
      const k6 = f(currentT + a6 * dt, currentY.map((val, i) => val + (b61 * k1[i] + b62 * k2[i] + b63 * k3[i] + b64 * k4[i] + b65 * k5[i]) * dt));

      const y4 = currentY.map((val, i) => 
        val + dt * (25 / 216 * k1[i] + 1408 / 2565 * k3[i] + 2197 / 4104 * k4[i] - 1 / 5 * k5[i])
      );
      
      const y5 = currentY.map((val, i) => 
        val + dt * (c1 * k1[i] + c3 * k3[i] + c4 * k4[i] + c5 * k5[i] + c6 * k6[i])
      );

      const error = Math.max(...y4.map((val, i) => Math.abs(val - y5[i])));
      
      if (error <= tol) {
        currentT += dt;
        currentY = y5;
        t.push(currentT);
        y.push([...currentY]);
      }

      h = dt * Math.min(5, Math.max(1e-2, 0.84 * Math.pow(tol / (error + 1e-10), 0.25)));
      steps++;
    }

    return { t, y };
  }

  private static bdf(
    f: VectorFunction,
    tSpan: [number, number],
    y0: Vector,
    h: number,
    maxSteps: number
  ): ODESolution {
    const t: number[] = [tSpan[0]];
    const y: number[][] = [[...y0]];
    let currentT = tSpan[0];
    let currentY = [...y0];
    let steps = 0;

    while (currentT < tSpan[1] && steps < maxSteps) {
      const dt = Math.min(h, tSpan[1] - currentT);
      const nextT = currentT + dt;
      
      let nextY = [...currentY];
      for (let iter = 0; iter < 10; iter++) {
        const dy = f(nextT, nextY);
        const residual = nextY.map((val, i) => val - currentY[i] - dt * dy[i]);
        const error = Math.max(...residual.map(Math.abs));
        
        if (error < 1e-8) break;
        
        nextY = nextY.map((val, i) => val - residual[i] / (1 - dt * this.approxJacobian(f, nextT, nextY, i)));
      }
      
      currentY = nextY;
      currentT = nextT;
      t.push(currentT);
      y.push([...currentY]);
      steps++;
    }

    return { t, y };
  }

  private static approxJacobian(
    f: VectorFunction,
    t: number,
    y: Vector,
    i: number,
    eps: number = 1e-8
  ): number {
    const yEps = [...y];
    yEps[i] += eps;
    const f1 = f(t, yEps);
    const f0 = f(t, y);
    return (f1[i] - f0[i]) / eps;
  }

  static lorenz(
    sigma: number = 10,
    rho: number = 28,
    beta: number = 8 / 3
  ): VectorFunction {
    return (t: number, y: Vector): Vector => {
      const [x, y_coord, z] = y;
      return [
        sigma * (y_coord - x),
        x * (rho - z) - y_coord,
        x * y_coord - beta * z
      ];
    };
  }
}

export function solve_ivp(
  fun: VectorFunction,
  t_span: [number, number],
  y0: Vector,
  options?: ODEOptions
): ODESolution {
  return ODESolver.solveIVP(fun, t_span, y0, options);
}

export function odeint(
  func: VectorFunction,
  y0: Vector,
  t: Vector,
  options?: ODEOptions
): ODESolution {
  const solution: ODESolution = { t: [t[0]], y: [[...y0]] };
  let currentY = [...y0];

  for (let i = 1; i < t.length; i++) {
    const result = ODESolver.solveIVP(func, [t[i - 1], t[i]], currentY, {
      ...options,
      method: 'rk4',
      stepSize: (t[i] - t[i - 1]) / 10
    });
    currentY = result.y[result.y.length - 1];
    solution.t.push(t[i]);
    solution.y.push([...currentY]);
  }

  return solution;
}

export function ode(
  f: VectorFunction,
  y0: Vector,
  t0: number,
  t1: number,
  options?: ODEOptions
): ODESolution {
  return ODESolver.solveIVP(f, [t0, t1], y0, options);
}
