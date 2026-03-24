import * as numcpp from '../src';

function testLinearAlgebra() {
  console.log('=== Linear Algebra Tests ===');
  
  const A = [[4, 2, 1], [2, 5, 3], [1, 3, 6]];
  const b = [7, 8, 9];
  
  console.log('Matrix A:', A);
  console.log('Vector b:', b);
  
  const x = numcpp.solve(A, b);
  console.log('Solution to Ax = b:', x);
  
  const det = numcpp.MatrixOps.determinant(A);
  console.log('Determinant of A:', det);
  
  const invA = numcpp.MatrixOps.inverse(A);
  console.log('Inverse of A:', invA);
  
  const { values, vectors } = numcpp.eig(A);
  console.log('Eigenvalues:', values);
  console.log('Eigenvectors:', vectors);
  
  const { U, S, V } = numcpp.svd(A);
  console.log('SVD - Singular values:', S);
  
  const A_rect = [[1, 2], [3, 4], [5, 6]];
  const b_rect = [5, 11, 17];
  const x_ls = numcpp.least_squares(A_rect, b_rect);
  console.log('Least squares solution:', x_ls);
  
  console.log();
}

function testIntegration() {
  console.log('=== Numerical Integration Tests ===');
  
  const f1 = (x: number) => x * x;
  const result1 = numcpp.quad(f1, 0, 1);
  console.log('Integral of x^2 from 0 to 1:', result1);
  console.log('Expected:', 1/3);
  
  const f2 = (x: number) => Math.sin(x);
  const result2 = numcpp.quad(f2, 0, Math.PI);
  console.log('Integral of sin(x) from 0 to π:', result2);
  console.log('Expected:', 2);
  
  const f3 = (x: number, y: number) => x * y;
  const result3 = numcpp.dblquad(f3, 0, 1, () => 0, () => 1);
  console.log('Double integral of xy from 0 to 1 for x and y:', result3);
  console.log('Expected:', 0.25);
  
  const x = [0, 1, 2, 3, 4, 5];
  const y = x.map(xi => xi * xi);
  const trapzResult = numcpp.trapz(x, y);
  console.log('Trapezoidal integral of discrete x^2:', trapzResult);
  
  console.log();
}

function testODE() {
  console.log('=== ODE Solver Tests ===');
  
  const simpleODE = (t: number, y: number[]) => [y[0]];
  const tSpan: [number, number] = [0, 5];
  const y0 = [1];
  
  const solution = numcpp.solve_ivp(simpleODE, tSpan, y0);
  console.log('Exponential growth ODE (dy/dt = y):');
  console.log('t:', solution.t.slice(0, 5));
  console.log('y:', solution.y.slice(0, 5).map(row => row[0]));
  console.log('Expected at t=5:', Math.exp(5));
  console.log('Computed at t=5:', solution.y[solution.y.length - 1][0]);
  
  const lorenz = numcpp.ODESolver.lorenz(10, 28, 8/3);
  const lorenzSolution = numcpp.solve_ivp(lorenz, [0, 50], [1, 1, 1], { method: 'rk45' });
  console.log('Lorenz attractor solution at t=50:', lorenzSolution.y[lorenzSolution.y.length - 1]);
  
  console.log();
}

function testNonlinear() {
  console.log('=== Nonlinear Solver Tests ===');
  
  const f1 = (x: number) => x * x - 2;
  const root1 = numcpp.root(f1, 0, 2);
  console.log('Root of x^2 - 2:', root1);
  console.log('Expected:', Math.sqrt(2));
  
  const brentRoot = numcpp.brentq(f1, 0, 2);
  console.log('Root using Brent\'s method:', brentRoot);
  
  const system = (x: number[]) => [
    x[0] + x[1] - 3,
    x[0] * x[0] + x[1] * x[1] - 9
  ];
  const x0 = [1, 1];
  const solution = numcpp.fsolve(system, x0);
  console.log('Solution to system x+y=3, x^2+y^2=9:', solution);
  
  const g = (x: number) => Math.cos(x);
  const fixedPoint = numcpp.NonLinearSolver.fixedPoint(g, 0);
  console.log('Fixed point of cos(x):', fixedPoint);
  
  console.log();
}

function testInterpolation() {
  console.log('=== Interpolation Tests ===');
  
  const x = [0, 1, 2, 3, 4, 5];
  const y = [0, 1, 4, 9, 16, 25];
  
  const linearInterp = numcpp.interp1d(x, y, { method: 'linear' });
  console.log('Linear interpolation at x=2.5:', linearInterp(2.5));
  
  const splineInterp = numcpp.interp1d(x, y, { method: 'spline' });
  console.log('Spline interpolation at x=2.5:', splineInterp(2.5));
  console.log('Expected:', 6.25);
  
  const tck = numcpp.splrep(x, y);
  const splevResult = numcpp.splev(2.5, tck);
  console.log('Spline evaluation at x=2.5 using splrep/splev:', splevResult);
  
  const x_grid = [0, 1, 2, 0, 1, 2];
  const y_grid = [0, 0, 0, 1, 1, 1];
  const z_grid = [0, 1, 4, 1, 2, 5];
  const xi = [0.5, 1.5];
  const yi = [0.5, 0.5];
  const gridResult = numcpp.griddata(x_grid, y_grid, z_grid, xi, yi);
  console.log('2D grid interpolation results:', gridResult);
  
  console.log();
}

function testSpecialFunctions() {
  console.log('=== Special Functions Tests ===');
  
  console.log('Gamma(5):', numcpp.gamma(5));
  console.log('Expected:', 24);
  
  console.log('Beta(2, 3):', numcpp.beta(2, 3));
  console.log('Expected:', 1/12);
  
  console.log('erf(1):', numcpp.erf(1));
  console.log('erfc(1):', numcpp.erfc(1));
  console.log('erfinv(erf(0.5)):', numcpp.erfinv(numcpp.erf(0.5)));
  
  console.log('Bessel J0(1):', numcpp.j0(1));
  console.log('Bessel J1(1):', numcpp.j1(1));
  console.log('Bessel Jn(2, 1):', numcpp.jn(2, 1));
  
  console.log('Legendre P2(0.5):', numcpp.legendre(2, 0.5));
  console.log('Expected:', -0.125);
  
  console.log('Hermite H3(1):', numcpp.hermite(3, 1));
  
  const [Ai, Bi] = numcpp.airy(0);
  console.log('Airy Ai(0):', Ai);
  console.log('Airy Bi(0):', Bi);
  
  console.log();
}

function main() {
  console.log('Testing NumCpp - Numerical Computation Library\n');
  
  testLinearAlgebra();
  testIntegration();
  testODE();
  testNonlinear();
  testInterpolation();
  testSpecialFunctions();
  
  console.log('=== All tests completed ===');
}

main();
