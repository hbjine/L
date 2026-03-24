import { Matrix, Vector } from '../core/types';
import { MatrixOps } from '../core/Matrix';
import { MatrixDecomposer } from './MatrixDecomposer';

export class LinearSystemSolver {
  static solve(A: Matrix, b: Vector): Vector {
    const n = A.length;
    
    if (n !== b.length) {
      throw new Error('Matrix and vector dimensions do not match');
    }

    const { L, U, P } = MatrixOps.luDecomposition(A);
    const Pb = MatrixOps.vectorMultiply(P, b);
    const y = MatrixOps.forwardSubstitution(L, Pb);
    const x = MatrixOps.backwardSubstitution(U, y);
    
    return x;
  }

  static leastSquares(A: Matrix, b: Vector): Vector {
    const m = A.length;
    const n = A[0].length;
    
    if (m !== b.length) {
      throw new Error('Matrix and vector dimensions do not match');
    }

    if (m >= n) {
      const { Q, R } = MatrixDecomposer.qr(A);
      if (!Q || !R) {
        throw new Error('QR decomposition failed');
      }
      
      const Qb = MatrixOps.vectorMultiply(MatrixOps.transpose(Q), b);
      const R1 = R.slice(0, n);
      const Qb1 = Qb.slice(0, n);
      
      return MatrixOps.backwardSubstitution(R1, Qb1);
    } else {
      const AT = MatrixOps.transpose(A);
      const ATA = MatrixOps.multiply(AT, A);
      const ATb = MatrixOps.vectorMultiply(AT, b);
      
      return this.solve(ATA, ATb);
    }
  }

  static solveTriangular(L: Matrix, b: Vector, lower: boolean = true): Vector {
    if (lower) {
      return MatrixOps.forwardSubstitution(L, b);
    } else {
      return MatrixOps.backwardSubstitution(L, b);
    }
  }

  static solvePositiveDefinite(A: Matrix, b: Vector): Vector {
    const { L } = MatrixDecomposer.cholesky(A);
    if (!L) {
      throw new Error('Cholesky decomposition failed');
    }
    
    const y = MatrixOps.forwardSubstitution(L, b);
    const x = MatrixOps.backwardSubstitution(MatrixOps.transpose(L), y);
    
    return x;
  }
}

export function solve(A: Matrix, b: Vector): Vector {
  return LinearSystemSolver.solve(A, b);
}

export function least_squares(A: Matrix, b: Vector): Vector {
  return LinearSystemSolver.leastSquares(A, b);
}
