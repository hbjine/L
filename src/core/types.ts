export type Matrix = number[][];
export type Vector = number[];
export type ScalarFunction = (x: number) => number;
export type VectorFunction = (t: number, y: Vector) => Vector;
export type MultiVariableFunction = (...args: number[]) => number;

export interface MatrixDecomposition {
  L?: Matrix;
  U?: Matrix;
  Q?: Matrix;
  R?: Matrix;
  P?: Matrix;
  D?: Matrix;
  V?: Matrix;
  S?: Matrix;
  singularValues?: Vector;
}

export interface EigenResult {
  values: Vector;
  vectors: Matrix;
}

export interface SVDResult {
  U: Matrix;
  S: Vector;
  V: Matrix;
}

export interface IntegrationOptions {
  method?: 'trapezoidal' | 'simpson' | 'gauss' | 'adaptive';
  tolerance?: number;
  maxIterations?: number;
  order?: number;
}

export interface ODESolution {
  t: Vector;
  y: Matrix;
}

export interface ODEOptions {
  method?: 'euler' | 'rk4' | 'rk45' | 'bdf';
  stepSize?: number;
  tolerance?: number;
  maxSteps?: number;
}

export interface RootFindingOptions {
  method?: 'bisection' | 'newton' | 'secant' | 'brentq';
  tolerance?: number;
  maxIterations?: number;
}

export interface InterpolationOptions {
  method?: 'linear' | 'nearest' | 'spline' | 'polynomial';
  order?: number;
  smooth?: number;
}
