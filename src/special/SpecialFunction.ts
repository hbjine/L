export class SpecialFunction {
  private static readonly GAMMA_COEFFICIENTS = [
    76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5
  ];

  static gamma(x: number): number {
    if (x <= 0 && Math.floor(x) === x) {
      return Infinity;
    }

    if (x < 0.5) {
      return Math.PI / (Math.sin(Math.PI * x) * this.gamma(1 - x));
    }

    x -= 1;
    let a = 1.000000000190015;
    const g = 5;

    for (let i = 0; i < this.GAMMA_COEFFICIENTS.length; i++) {
      a += this.GAMMA_COEFFICIENTS[i] / (x + i + 1);
    }

    const tmp = x + g + 0.5;
    return Math.sqrt(2 * Math.PI) * Math.pow(tmp, x + 0.5) * Math.exp(-tmp) * a;
  }

  static logGamma(x: number): number {
    if (x <= 0) {
      return Infinity;
    }

    if (x < 0.5) {
      return Math.log(Math.PI / Math.sin(Math.PI * x)) - this.logGamma(1 - x);
    }

    x -= 1;
    let a = 1.000000000190015;
    const g = 5;

    for (let i = 0; i < this.GAMMA_COEFFICIENTS.length; i++) {
      a += this.GAMMA_COEFFICIENTS[i] / (x + i + 1);
    }

    const tmp = x + g + 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(tmp) - tmp + Math.log(a);
  }

  static beta(z: number, w: number): number {
    return Math.exp(this.logGamma(z) + this.logGamma(w) - this.logGamma(z + w));
  }

  static factorial(n: number): number {
    if (n < 0 || Math.floor(n) !== n) {
      throw new Error('Factorial is only defined for non-negative integers');
    }
    return this.gamma(n + 1);
  }

  static erf(x: number): number {
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;

    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x);

    const t = 1 / (1 + p * x);
    const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

    return sign * y;
  }

  static erfc(x: number): number {
    return 1 - this.erf(x);
  }

  static erfinv(x: number): number {
    if (x <= -1 || x >= 1) {
      throw new Error('Argument must be between -1 and 1');
    }

    if (Math.abs(x) === 1) {
      return x === 1 ? Infinity : -Infinity;
    }

    const a = 0.140012;
    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x);

    const part1 = Math.log(1 - x * x);
    const part2 = 2 / (Math.PI * a) + part1 / 2;
    const part3 = part2 * part2 - part1 / a;
    const part4 = Math.sqrt(part3);
    const part5 = part4 - part2;

    return sign * Math.sqrt(part5);
  }

  static besselJ0(x: number): number {
    const ax = Math.abs(x);
    
    if (ax < 8) {
      const y = x * x;
      const ans1 = 57568490574 + y * (-13362590354 + y * (651619640.7 + 
        y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
      const ans2 = 57568490411 + y * (1029532985 + y * (9494680.718 + 
        y * (59272.64853 + y * (267.8532712 + y * 1))));
      return ans1 / ans2;
    } else {
      const z = 8 / ax;
      const y = z * z;
      const xx = ax - 2.356194491;
      const ans1 = 1 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + 
        y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
      const ans2 = -0.1562499995e-1 + y * (0.1430488765e-3 + 
        y * (-0.6911147651e-5 + y * (0.7621095161e-6 + y * (-0.934945152e-7))));
      return Math.sqrt(0.636619772 / ax) * (Math.cos(xx) * ans1 - z * Math.sin(xx) * ans2);
    }
  }

  static besselJ1(x: number): number {
    const ax = Math.abs(x);
    
    if (ax < 8) {
      const y = x * x;
      const ans1 = x * (72362614232 + y * (-7895059235 + y * (242396853.1 + 
        y * (-2972611.439 + y * (15704.4826 + y * (-30.16036606))))));
      const ans2 = 144725228442 + y * (2300535178 + y * (18583304.74 + 
        y * (99447.43394 + y * (376.9991397 + y * 1))));
      return ans1 / ans2;
    } else {
      const z = 8 / ax;
      const y = z * z;
      const xx = ax - 3.926990817;
      const ans1 = 1 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + 
        y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
      const ans2 = -0.1562499995e-1 + y * (0.1430488765e-3 + 
        y * (-0.6911147651e-5 + y * (0.7621095161e-6 + y * (-0.934945152e-7))));
      const result = Math.sqrt(0.636619772 / ax) * (Math.cos(xx) * ans1 - z * Math.sin(xx) * ans2);
      return x < 0 ? -result : result;
    }
  }

  static besselJn(n: number, x: number): number {
    if (n === 0) return this.besselJ0(x);
    if (n === 1) return this.besselJ1(x);
    
    if (x === 0) return 0;
    
    const ax = Math.abs(x);
    const sign = x < 0 && n % 2 === 1 ? -1 : 1;
    
    if (ax > n) {
      let j0 = this.besselJ0(ax);
      let j1 = this.besselJ1(ax);
      
      for (let m = 1; m < n; m++) {
        const j2 = 2 * m * j1 / ax - j0;
        j0 = j1;
        j1 = j2;
      }
      
      return sign * j1;
    } else {
      const m = Math.floor((Math.floor((n + Math.sqrt(40 * n)) / 2) + n) / 2) + 15;
      let j0 = 0;
      let j1 = 1;
      let jsum = 0;
      let sum = 0;
      
      for (let k = m; k >= 0; k--) {
        const j2 = 2 * (k + 1) * j1 / ax - j0;
        j0 = j1;
        j1 = j2;
        jsum += j0;
        
        if (k === n) sum = j0;
        if (k === 0) jsum = 2 * jsum - j0;
      }
      
      return sign * sum / jsum;
    }
  }

  static hypergeometric0F1(a: number, x: number, maxIter: number = 100, tol: number = 1e-10): number {
    if (a <= 0) throw new Error('Parameter a must be positive');
    
    let term = 1;
    let sum = term;
    
    for (let n = 0; n < maxIter; n++) {
      term *= x / ((a + n) * (n + 1));
      sum += term;
      if (Math.abs(term) < tol) break;
    }
    
    return sum;
  }

  static hypergeometric1F1(a: number, b: number, x: number, maxIter: number = 100, tol: number = 1e-10): number {
    if (b <= 0) throw new Error('Parameter b must be positive');
    
    let term = 1;
    let sum = term;
    
    for (let n = 0; n < maxIter; n++) {
      term *= (a + n) * x / ((b + n) * (n + 1));
      sum += term;
      if (Math.abs(term) < tol) break;
    }
    
    return sum;
  }

  static hypergeometric2F1(a: number, b: number, c: number, x: number, maxIter: number = 100, tol: number = 1e-10): number {
    if (Math.abs(x) > 1) throw new Error('|x| must be <= 1 for convergence');
    if (c <= 0) throw new Error('Parameter c must be positive');
    
    let term = 1;
    let sum = term;
    
    for (let n = 0; n < maxIter; n++) {
      term *= (a + n) * (b + n) * x / ((c + n) * (n + 1));
      sum += term;
      if (Math.abs(term) < tol) break;
    }
    
    return sum;
  }

  static airyAi(x: number): number {
    const absX = Math.abs(x);
    const sqrtX = Math.sqrt(absX);
    
    if (x > 0) {
      const z = (2 / 3) * Math.pow(absX, 1.5);
      const k = (1 / 3);
      const besselK13 = this.modifiedBesselK(k, z);
      const besselK23 = this.modifiedBesselK(2 * k, z);
      return (sqrtX / Math.PI) * Math.sqrt(1 / 3) * (besselK23 - besselK13);
    } else if (x < 0) {
      const z = (2 / 3) * Math.pow(absX, 1.5);
      const k = (1 / 3);
      return (sqrtX / 3) * (this.besselJn(-k, z) + this.besselJn(k, z));
    } else {
      return 1 / (Math.pow(3, 2 / 3) * this.gamma(2 / 3));
    }
  }

  static airyBi(x: number): number {
    const absX = Math.abs(x);
    const sqrtX = Math.sqrt(absX);
    
    if (x > 0) {
      const z = (2 / 3) * Math.pow(absX, 1.5);
      const k = (1 / 3);
      const besselI13 = this.modifiedBesselI(k, z);
      const besselI23 = this.modifiedBesselI(2 * k, z);
      return sqrtX * Math.sqrt(1 / 3) * (besselI13 + besselI23);
    } else if (x < 0) {
      const z = (2 / 3) * Math.pow(absX, 1.5);
      const k = (1 / 3);
      return sqrtX * (this.besselJn(-k, z) - this.besselJn(k, z));
    } else {
      return 1 / (Math.pow(3, 1 / 6) * this.gamma(2 / 3));
    }
  }

  private static modifiedBesselI(nu: number, x: number): number {
    if (x < 0) throw new Error('x must be non-negative');
    if (x === 0) return nu === 0 ? 1 : 0;
    
    const halfX = x / 2;
    let sum = 0;
    
    for (let k = 0; k < 50; k++) {
      const term = Math.pow(halfX, 2 * k + nu) / (this.gamma(k + 1) * this.gamma(k + nu + 1));
      sum += term;
      if (term < 1e-15) break;
    }
    
    return sum;
  }

  private static modifiedBesselK(nu: number, x: number): number {
    if (x <= 0) throw new Error('x must be positive');
    
    const piOver2 = Math.PI / 2;
    const iNu = this.modifiedBesselI(nu, x);
    const iNegNu = this.modifiedBesselI(-nu, x);
    
    return piOver2 * (iNegNu - iNu) / Math.sin(nu * Math.PI);
  }

  static legendreP(n: number, x: number): number {
    if (n < 0 || Math.floor(n) !== n) {
      throw new Error('n must be a non-negative integer');
    }
    
    if (Math.abs(x) > 1) throw new Error('|x| must be <= 1');
    
    if (n === 0) return 1;
    if (n === 1) return x;
    
    let p0 = 1;
    let p1 = x;
    
    for (let k = 1; k < n; k++) {
      const p2 = ((2 * k + 1) * x * p1 - k * p0) / (k + 1);
      p0 = p1;
      p1 = p2;
    }
    
    return p1;
  }

  static hermiteH(n: number, x: number): number {
    if (n < 0 || Math.floor(n) !== n) {
      throw new Error('n must be a non-negative integer');
    }
    
    if (n === 0) return 1;
    if (n === 1) return 2 * x;
    
    let h0 = 1;
    let h1 = 2 * x;
    
    for (let k = 1; k < n; k++) {
      const h2 = 2 * x * h1 - 2 * k * h0;
      h0 = h1;
      h1 = h2;
    }
    
    return h1;
  }

  static laguerreL(n: number, x: number): number {
    if (n < 0 || Math.floor(n) !== n) {
      throw new Error('n must be a non-negative integer');
    }
    
    if (n === 0) return 1;
    if (n === 1) return 1 - x;
    
    let l0 = 1;
    let l1 = 1 - x;
    
    for (let k = 1; k < n; k++) {
      const l2 = ((2 * k + 1 - x) * l1 - k * l0) / (k + 1);
      l0 = l1;
      l1 = l2;
    }
    
    return l1;
  }

  static chebyshevT(n: number, x: number): number {
    if (n < 0 || Math.floor(n) !== n) {
      throw new Error('n must be a non-negative integer');
    }
    
    if (Math.abs(x) > 1) throw new Error('|x| must be <= 1');
    
    return Math.cos(n * Math.acos(x));
  }

  static chebyshevU(n: number, x: number): number {
    if (n < 0 || Math.floor(n) !== n) {
      throw new Error('n must be a non-negative integer');
    }
    
    if (Math.abs(x) > 1) throw new Error('|x| must be <= 1');
    
    const theta = Math.acos(x);
    return Math.sin((n + 1) * theta) / Math.sin(theta);
  }
}

export function gamma(x: number): number {
  return SpecialFunction.gamma(x);
}

export function gammaln(x: number): number {
  return SpecialFunction.logGamma(x);
}

export function beta(z: number, w: number): number {
  return SpecialFunction.beta(z, w);
}

export function factorial(n: number): number {
  return SpecialFunction.factorial(n);
}

export function erf(x: number): number {
  return SpecialFunction.erf(x);
}

export function erfc(x: number): number {
  return SpecialFunction.erfc(x);
}

export function erfinv(x: number): number {
  return SpecialFunction.erfinv(x);
}

export function j0(x: number): number {
  return SpecialFunction.besselJ0(x);
}

export function j1(x: number): number {
  return SpecialFunction.besselJ1(x);
}

export function jn(n: number, x: number): number {
  return SpecialFunction.besselJn(n, x);
}

export function hyp0f1(a: number, x: number): number {
  return SpecialFunction.hypergeometric0F1(a, x);
}

export function hyp1f1(a: number, b: number, x: number): number {
  return SpecialFunction.hypergeometric1F1(a, b, x);
}

export function hyp2f1(a: number, b: number, c: number, x: number): number {
  return SpecialFunction.hypergeometric2F1(a, b, c, x);
}

export function airy(x: number): [number, number] {
  return [SpecialFunction.airyAi(x), SpecialFunction.airyBi(x)];
}

export function legendre(n: number, x: number): number {
  return SpecialFunction.legendreP(n, x);
}

export function hermite(n: number, x: number): number {
  return SpecialFunction.hermiteH(n, x);
}

export function laguerre(n: number, x: number): number {
  return SpecialFunction.laguerreL(n, x);
}

export function chebyt(n: number, x: number): number {
  return SpecialFunction.chebyshevT(n, x);
}

export function chebyu(n: number, x: number): number {
  return SpecialFunction.chebyshevU(n, x);
}
