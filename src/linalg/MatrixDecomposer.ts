import { Matrix, Vector, MatrixDecomposition, EigenResult, SVDResult } from '../core/types';
import { MatrixOps } from '../core/Matrix';

export class MatrixDecomposer {
  static lu(A: Matrix): MatrixDecomposition {
    return MatrixOps.luDecomposition(A);
  }

  static qr(A: Matrix): MatrixDecomposition {
    const m = A.length;
    const n = A[0].length;
    const Q = MatrixOps.copy(A);
    const R = MatrixOps.zeros(n, n);

    for (let j = 0; j < n; j++) {
      let v = MatrixOps.column(Q, j);
      let norm = MatrixOps.norm(v);
      R[j][j] = norm;
      
      v = v.map(val => val / norm);
      MatrixOps.setColumn(Q, j, v);

      for (let k = j + 1; k < n; k++) {
        const colK = MatrixOps.column(Q, k);
        const dot = v.reduce((sum, val, i) => sum + val * colK[i], 0);
        R[j][k] = dot;
        
        for (let i = 0; i < m; i++) {
          Q[i][k] -= dot * v[i];
        }
      }
    }

    return { Q: Q.slice(0, n), R };
  }

  static cholesky(A: Matrix): MatrixDecomposition {
    const n = A.length;
    const L = MatrixOps.zeros(n, n);

    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = A[i][j];
        for (let k = 0; k < j; k++) {
          sum -= L[i][k] * L[j][k];
        }
        if (i === j) {
          L[i][j] = Math.sqrt(sum);
        } else {
          L[i][j] = sum / L[j][j];
        }
      }
    }

    return { L };
  }

  static svd(A: Matrix): SVDResult {
    const m = A.length;
    const n = A[0].length;
    const AT = MatrixOps.transpose(A);
    const ATA = MatrixOps.multiply(AT, A);
    const eigResult = this.eig(ATA);
    
    const sortedIndices = eigResult.values
      .map((val, idx) => ({ val, idx }))
      .sort((a, b) => b.val - a.val)
      .map(item => item.idx);

    const S = sortedIndices
      .map(i => Math.sqrt(Math.max(0, eigResult.values[i])))
      .filter(val => val > 1e-10);

    const V = MatrixOps.zeros(n, S.length);
    for (let i = 0; i < S.length; i++) {
      const col = MatrixOps.column(eigResult.vectors, sortedIndices[i]);
      const norm = MatrixOps.norm(col);
      for (let j = 0; j < n; j++) {
        V[j][i] = col[j] / norm;
      }
    }

    const U = MatrixOps.zeros(m, S.length);
    for (let i = 0; i < S.length; i++) {
      const vCol = MatrixOps.column(V, i);
      const av = MatrixOps.vectorMultiply(A, vCol);
      for (let j = 0; j < m; j++) {
        U[j][i] = av[j] / S[i];
      }
    }

    return { U, S, V };
  }

  static eig(A: Matrix): EigenResult {
    const n = A.length;
    let X = MatrixOps.copy(A);
    const maxIter = 100;
    const tol = 1e-10;

    const eigenVectors = MatrixOps.identity(n);

    for (let iter = 0; iter < maxIter; iter++) {
      const { Q, R } = this.qr(X);
      if (!Q || !R) break;
      
      X = MatrixOps.multiply(R, Q);
      const newEigVec = MatrixOps.multiply(eigenVectors, Q);
      
      let converged = true;
      for (let i = 0; i < n && converged; i++) {
        for (let j = 0; j < i && converged; j++) {
          if (Math.abs(X[i][j]) > tol) {
            converged = false;
          }
        }
      }
      
      if (converged) break;
    }

    const values = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      values[i] = X[i][i];
    }

    return { values, vectors: eigenVectors };
  }
}

export function eig(A: Matrix): EigenResult {
  return MatrixDecomposer.eig(A);
}

export function svd(A: Matrix): SVDResult {
  return MatrixDecomposer.svd(A);
}
