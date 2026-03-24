import { Matrix, Vector } from './types';

export class MatrixOps {
  static create(rows: number, cols: number, fill: number = 0): Matrix {
    return Array(rows).fill(null).map(() => Array(cols).fill(fill));
  }

  static identity(n: number): Matrix {
    const I = this.create(n, n);
    for (let i = 0; i < n; i++) {
      I[i][i] = 1;
    }
    return I;
  }

  static zeros(rows: number, cols: number): Matrix {
    return this.create(rows, cols, 0);
  }

  static ones(rows: number, cols: number): Matrix {
    return this.create(rows, cols, 1);
  }

  static copy(A: Matrix): Matrix {
    return A.map(row => [...row]);
  }

  static transpose(A: Matrix): Matrix {
    const rows = A.length;
    const cols = A[0].length;
    const AT = this.create(cols, rows);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        AT[j][i] = A[i][j];
      }
    }
    return AT;
  }

  static add(A: Matrix, B: Matrix): Matrix {
    const rows = A.length;
    const cols = A[0].length;
    const C = this.create(rows, cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        C[i][j] = A[i][j] + B[i][j];
      }
    }
    return C;
  }

  static subtract(A: Matrix, B: Matrix): Matrix {
    const rows = A.length;
    const cols = A[0].length;
    const C = this.create(rows, cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        C[i][j] = A[i][j] - B[i][j];
      }
    }
    return C;
  }

  static multiply(A: Matrix, B: Matrix): Matrix {
    const m = A.length;
    const n = B[0].length;
    const p = B.length;
    const C = this.create(m, n);
    for (let i = 0; i < m; i++) {
      for (let k = 0; k < p; k++) {
        if (A[i][k] !== 0) {
          for (let j = 0; j < n; j++) {
            C[i][j] += A[i][k] * B[k][j];
          }
        }
      }
    }
    return C;
  }

  static scalarMultiply(A: Matrix, scalar: number): Matrix {
    return A.map(row => row.map(val => val * scalar));
  }

  static vectorMultiply(A: Matrix, x: Vector): Vector {
    const m = A.length;
    const n = A[0].length;
    const b = new Array(m).fill(0);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        b[i] += A[i][j] * x[j];
      }
    }
    return b;
  }

  static norm(x: Vector): number {
    return Math.sqrt(x.reduce((sum, val) => sum + val * val, 0));
  }

  static column(A: Matrix, j: number): Vector {
    return A.map(row => row[j]);
  }

  static setColumn(A: Matrix, j: number, col: Vector): void {
    for (let i = 0; i < A.length; i++) {
      A[i][j] = col[i];
    }
  }

  static diag(v: Vector): Matrix {
    const n = v.length;
    const D = this.zeros(n, n);
    for (let i = 0; i < n; i++) {
      D[i][i] = v[i];
    }
    return D;
  }

  static trace(A: Matrix): number {
    let sum = 0;
    for (let i = 0; i < A.length; i++) {
      sum += A[i][i];
    }
    return sum;
  }

  static luDecomposition(A: Matrix): { L: Matrix; U: Matrix; P: Matrix } {
    const n = A.length;
    const L = this.identity(n);
    const U = this.copy(A);
    const P = this.identity(n);

    for (let k = 0; k < n - 1; k++) {
      let pivotRow = k;
      let maxVal = Math.abs(U[k][k]);
      for (let i = k + 1; i < n; i++) {
        if (Math.abs(U[i][k]) > maxVal) {
          maxVal = Math.abs(U[i][k]);
          pivotRow = i;
        }
      }

      if (pivotRow !== k) {
        [U[k], U[pivotRow]] = [U[pivotRow], U[k]];
        [P[k], P[pivotRow]] = [P[pivotRow], P[k]];
        for (let i = 0; i < k; i++) {
          [L[k][i], L[pivotRow][i]] = [L[pivotRow][i], L[k][i]];
        }
      }

      for (let i = k + 1; i < n; i++) {
        L[i][k] = U[i][k] / U[k][k];
        for (let j = k; j < n; j++) {
          U[i][j] -= L[i][k] * U[k][j];
        }
      }
    }

    return { L, U, P };
  }

  static forwardSubstitution(L: Matrix, b: Vector): Vector {
    const n = L.length;
    const y = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      y[i] = b[i];
      for (let j = 0; j < i; j++) {
        y[i] -= L[i][j] * y[j];
      }
      y[i] /= L[i][i];
    }
    return y;
  }

  static backwardSubstitution(U: Matrix, y: Vector): Vector {
    const n = U.length;
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      x[i] = y[i];
      for (let j = i + 1; j < n; j++) {
        x[i] -= U[i][j] * x[j];
      }
      x[i] /= U[i][i];
    }
    return x;
  }

  static determinant(A: Matrix): number {
    const { U, P } = this.luDecomposition(A);
    let det = 1;
    let sign = 1;
    for (let i = 0; i < P.length; i++) {
      if (P[i][i] !== 1) sign *= -1;
    }
    for (let i = 0; i < U.length; i++) {
      det *= U[i][i];
    }
    return det * sign;
  }

  static inverse(A: Matrix): Matrix {
    const n = A.length;
    const { L, U, P } = this.luDecomposition(A);
    const inv = this.zeros(n, n);

    for (let j = 0; j < n; j++) {
      const ej = P.map(row => row[j]);
      const y = this.forwardSubstitution(L, ej);
      const x = this.backwardSubstitution(U, y);
      this.setColumn(inv, j, x);
    }

    return inv;
  }
}
