/**
 * A compact GPT-style decoder-only transformer implemented from scratch in
 * plain TypeScript (no external numerics library).
 *
 * Forward and backward passes are written explicitly for every component:
 * token embedding (+ tied output head), learned positional embeddings,
 * RMSNorm, causal multi-head self-attention, and GeLU MLP. This keeps the
 * model fully transparent and runnable in a browser or Node without any
 * external dependencies.
 *
 * All parameters are Float64Array with a matching gradient store. The public
 * API exposes weights/grads so an arbitrary optimizer can drive training.
 */

import { defaultGPTConfig, type GPTConfig, type TrainingConfig } from "./config.js";

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);

/** Exact GeLU activation (GPT-2 style tanh approximation). */
export function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)));
}

function geluDerivative(x: number): number {
  const u = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
  const tanh = Math.tanh(u);
  const du = SQRT_2_OVER_PI * (1 + 3 * 0.044715 * x * x);
  return 0.5 * (1 + tanh) + 0.5 * x * (1 - tanh * tanh) * du;
}

interface RMSNormCache {
  /** Copy of the pre-normalized input (needed for backward). */
  x: Float64Array;
  /** Per-element reciprocal stddev, repeated over the row. */
  r: Float64Array;
}

/**
 * RMSNorm: y = x / sqrt(mean(x^2) + eps) * gamma.
 * `cache.x` must be length C*T and holds copies of the input rows; `cache.r`
 * is parallel (the normalizer for each row element).
 */
export function rmsnorm(
  x: Float64Array,
  gamma: Float64Array,
  eps: number,
  out: Float64Array,
  cache: RMSNormCache,
): void {
  const dim = x.length;
  let sum = 0;
  for (let i = 0; i < dim; i++) sum += x[i] * x[i];
  const r = 1 / Math.sqrt(sum / dim + eps);
  for (let i = 0; i < dim; i++) {
    cache.x[i] = x[i];
    cache.r[i] = r;
    out[i] = x[i] * r * gamma[i];
  }
}

/**
 * Backward for RMSNorm. Given dy for the output row, writes dx to `out` and
 * accumulates the gamma gradient into `dgamma`.
 */
export function rmsnormBackward(
  dy: Float64Array,
  gamma: Float64Array,
  dgamma: Float64Array,
  cache: RMSNormCache,
  out: Float64Array,
): void {
  const dim = dy.length;
  let sumDyhatNorm = 0;
  for (let i = 0; i < dim; i++) {
    const yhat = cache.x[i] * cache.r[i];
    const dyhat = dy[i] * gamma[i];
    sumDyhatNorm += dyhat * yhat;
  }
  const meanDyhatNorm = sumDyhatNorm / dim;
  for (let i = 0; i < dim; i++) {
    const yhat = cache.x[i] * cache.r[i];
    const dyhat = dy[i] * gamma[i];
    out[i] = (dyhat - yhat * meanDyhatNorm) * cache.r[i];
    dgamma[i] += dy[i] * yhat;
  }
}

/**
 * GPT language model.
 *
 * Weight layout (all row-major Float64Array):
 *   wte   [vocabSize * nEmb]  token embedding (also the output head)
 *   wpe   [blockSize * nEmb]  positional embedding
 *   per block: anorm[nEmb], wqkv[nEmb * 3*nEmb], wout[nEmb*nEmb],
 *              mnorm[nEmb], w1[nEmb * 4*nEmb], w2[4*nEmb * nEmb]
 *   fnorm [nEmb]
 *
 * Activations use row-major (T*C); the per-head projections internally use a
 * head-major layout (H*T*HD) that is consistent between forward and backward.
 */
export class GPT {
  readonly cfg: GPTConfig;
  readonly v: number;
  readonly c: number;
  readonly nHead: number;
  readonly headDim: number;
  readonly mlpDim: number;

  readonly weights: Map<string, Float64Array> = new Map();
  readonly grads: Map<string, Float64Array> = new Map();
  /** Fixed iteration order of parameters for the optimizer. */
  readonly order: string[] = [];

  constructor(config: Partial<GPTConfig>) {
    const cfg = defaultGPTConfig(config);
    if (cfg.nEmb % cfg.nHead !== 0) throw new Error("nEmb must be divisible by nHead");
    if (cfg.nLayer < 1 || cfg.blockSize < 1 || cfg.vocabSize < 1) throw new Error("invalid dims");
    this.cfg = cfg;
    this.v = cfg.vocabSize;
    this.c = cfg.nEmb;
    this.nHead = cfg.nHead;
    this.headDim = cfg.nEmb / cfg.nHead;
    this.mlpDim = cfg.nEmb * (cfg.mlpScale ?? 4);

    this.addParam("wte", cfg.vocabSize * cfg.nEmb);
    this.addParam("wpe", cfg.blockSize * cfg.nEmb);
    for (let l = 0; l < cfg.nLayer; l++) {
      this.addParam(`b${l}.anorm`, cfg.nEmb);
      this.addParam(`b${l}.wqkv`, cfg.nEmb * 3 * cfg.nEmb);
      this.addParam(`b${l}.wout`, cfg.nEmb * cfg.nEmb);
      this.addParam(`b${l}.mnorm`, cfg.nEmb);
      this.addParam(`b${l}.w1`, cfg.nEmb * this.mlpDim);
      this.addParam(`b${l}.w2`, this.mlpDim * cfg.nEmb);
    }
    this.addParam("fnorm", cfg.nEmb);
    this.init();
  }

  private addParam(name: string, size: number): void {
    this.weights.set(name, new Float64Array(size));
    this.grads.set(name, new Float64Array(size));
    this.order.push(name);
  }

  /** GPT-2 style init: N(0, 0.02), residual-out weights scaled by 1/sqrt(2L). */
  init(): void {
    const base = 0.02;
    for (const name of this.order) {
      const w = this.weights.get(name)!;
      const residualOut = name.endsWith(".wout") || name.endsWith(".w2");
      const s = residualOut ? base / Math.sqrt(2 * this.cfg.nLayer) : base;
      for (let i = 0; i < w.length; i++) w[i] = gauss() * s;
    }
    for (const name of this.order) {
      if (name.endsWith(".anorm") || name.endsWith(".mnorm") || name === "fnorm") {
        this.weights.get(name)!.fill(1);
      }
    }
    this.zeroGrads();
  }

  zeroGrads(): void {
    for (const name of this.order) this.grads.get(name)!.fill(0);
  }

  get paramCount(): number {
    let n = 0;
    for (const name of this.order) n += this.weights.get(name)!.length;
    return n;
  }

  private eps(): number {
    return this.cfg.normEps ?? 1e-5;
  }

  // ================= FORWARD =================

  /**
   * Forward over a batch. `tokens` is an Int32Array of B*T ids (row-major),
   * `targets` the same shape. Returns averaged loss and per-item caches.
   */
  forward(tokens: Int32Array, targets: Int32Array): { loss: number; caches: ItemCache[] } {
    const T = this.cfg.blockSize;
    const B = targets.length / T;
    if (!Number.isInteger(B) || tokens.length !== targets.length || B === 0) {
      throw new Error("targets must be B*T tokens matching tokens length, B>0");
    }
    const caches: ItemCache[] = [];
    let total = 0;
    for (let b = 0; b < B; b++) {
      const cache = this.forwardItem(tokens, targets, b);
      caches.push(cache);
      total += cache.loss;
    }
    return { loss: total / B, caches };
  }

  /** Backward, scaling gradients by 1/B to match the averaged loss. */
  backward(tokens: Int32Array, targets: Int32Array, caches: ItemCache[]): void {
    const T = this.cfg.blockSize;
    const B = targets.length / T;
    for (let b = 0; b < B; b++) this.backwardItem(tokens, targets, b, caches[b]);
  }

  private embedForward(tokens: Int32Array, b: number, cache: ItemCache): void {
    const { c: C } = this;
    const T = this.cfg.blockSize;
    const wte = this.weights.get("wte")!;
    const wpe = this.weights.get("wpe")!;
    for (let t = 0; t < T; t++) {
      const id = tokens[b * T + t];
      const base = t * C;
      for (let i = 0; i < C; i++) {
        cache.x[base + i] = wte[id * C + i] + wpe[base + i];
      }
    }
  }

  private forwardItem(tokens: Int32Array, targets: Int32Array, b: number): ItemCache {
    const { v: V, c: C, nHead: H, headDim: HD } = this;
    const T = this.cfg.blockSize;
    const cache: ItemCache = {
      x: new Float64Array(C * T),
      loss: 0,
      perLayer: [],
      ln: new Float64Array(C * T),
      fln: { x: new Float64Array(C * T), r: new Float64Array(C * T) },
      logits: new Float64Array(V * T),
    };
    this.embedForward(tokens, b, cache);

    let x = cache.x;
    for (let l = 0; l < this.cfg.nLayer; l++) {
      const lc: LayerCache = {
        xn1: new Float64Array(C * T),
        an: { x: new Float64Array(C * T), r: new Float64Array(C * T) },
        ar1: new Float64Array(C * T),
        qh: new Float64Array(H * T * HD),
        kh: new Float64Array(H * T * HD),
        vh: new Float64Array(H * T * HD),
        probs: new Float64Array(H * T * T),
        aPre: new Float64Array(C * T),
        a: new Float64Array(C * T),
        xn2: new Float64Array(C * T),
        mn: { x: new Float64Array(C * T), r: new Float64Array(C * T) },
        pre: new Float64Array(this.mlpDim * T),
        hpost: new Float64Array(this.mlpDim * T),
        m: new Float64Array(C * T),
        out: new Float64Array(C * T),
      };
      this.forwardLayer(l, x, lc);
      x = lc.out;
      cache.perLayer.push(lc);
    }

    // final norm + tied head
    const fnorm = this.weights.get("fnorm")!;
    const wte = this.weights.get("wte")!;
    let sumLoss = 0;
    for (let t = 0; t < T; t++) {
      const off = t * C;
      rmsnorm(x.subarray(off, off + C), fnorm, this.eps(), cache.ln.subarray(off, off + C) as Float64Array, {
        x: cache.fln.x.subarray(off, off + C) as Float64Array,
        r: cache.fln.r.subarray(off, off + C) as Float64Array,
      });

      let maxL = -Infinity;
      for (let j = 0; j < V; j++) {
        let s = 0;
        for (let i = 0; i < C; i++) s += cache.ln[off + i] * wte[j * C + i];
        cache.logits[t * V + j] = s;
        if (s > maxL) maxL = s;
      }
      let sumExp = 0;
      for (let j = 0; j < V; j++) sumExp += Math.exp(cache.logits[t * V + j] - maxL);
      const logZ = Math.log(sumExp) + maxL;
      const y = targets[b * T + t];
      sumLoss += logZ - cache.logits[t * V + y];
    }
    cache.loss = sumLoss / T;
    return cache;
  }

  private forwardLayer(l: number, x: Float64Array, lc: LayerCache): void {
    const { c: C, nHead: H, headDim: HD, mlpDim: MLP } = this;
    const T = this.cfg.blockSize;
    const anorm = this.weights.get(`b${l}.anorm`)!;
    const wqkv = this.weights.get(`b${l}.wqkv`)!;
    const wout = this.weights.get(`b${l}.wout`)!;
    const mnorm = this.weights.get(`b${l}.mnorm`)!;
    const w1 = this.weights.get(`b${l}.w1`)!;
    const w2 = this.weights.get(`b${l}.w2`)!;
    const eps = this.eps();
    const scale = 1 / Math.sqrt(HD);

    // rmsnorm 1
    for (let t = 0; t < T; t++) {
      const off = t * C;
      rmsnorm(x.subarray(off, off + C), anorm, eps, lc.xn1.subarray(off, off + C) as Float64Array, {
        x: lc.an.x.subarray(off, off + C) as Float64Array,
        r: lc.an.r.subarray(off, off + C) as Float64Array,
      });
    }

    // qkv projection -> head-major q/k/v
    for (let t = 0; t < T; t++) {
      for (let o = 0; o < 3 * C; o++) {
        let s = 0;
        for (let i = 0; i < C; i++) s += lc.xn1[t * C + i] * wqkv[o * C + i];
        const dst = o < C ? lc.qh : o < 2 * C ? lc.kh : lc.vh;
        const col = o % C;
        const h = Math.floor(col / HD);
        const d = col % HD;
        dst[h * (T * HD) + t * HD + d] = s;
      }
    }

    // attention: raw causal scores stored in probs buffer, softmax in place
    for (let h = 0; h < H; h++) {
      const hb = h * (T * HD);
      const rowBase = h * T * T;
      for (let t = 0; t < T; t++) {
        for (let t2 = 0; t2 <= t; t2++) {
          let s = 0;
          for (let d = 0; d < HD; d++) s += lc.qh[hb + t * HD + d] * lc.kh[hb + t2 * HD + d];
          lc.probs[rowBase + t * T + t2] = s * scale;
        }
      }
    }
    // softmax rows
    for (let h = 0; h < H; h++) {
      const rowBase = h * T * T;
      for (let t = 0; t < T; t++) {
        let maxS = -Infinity;
        for (let t2 = 0; t2 <= t; t2++) {
          const v = lc.probs[rowBase + t * T + t2];
          if (v > maxS) maxS = v;
        }
        let sumExp = 0;
        for (let t2 = 0; t2 <= t; t2++) {
          const e = Math.exp(lc.probs[rowBase + t * T + t2] - maxS);
          lc.probs[rowBase + t * T + t2] = e;
          sumExp += e;
        }
        for (let t2 = 0; t2 <= t; t2++) lc.probs[rowBase + t * T + t2] /= sumExp;
      }
    }
    // context: aPre (row-major) += sum_t2 p*v
    for (let h = 0; h < H; h++) {
      const hb = h * (T * HD);
      const colBase = h * HD;
      const rowBase = h * T * T;
      for (let t = 0; t < T; t++) {
        for (let t2 = 0; t2 <= t; t2++) {
          const p = lc.probs[rowBase + t * T + t2];
          if (p === 0) continue;
          for (let d = 0; d < HD; d++) {
            lc.aPre[t * C + colBase + d] += p * lc.vh[hb + t2 * HD + d];
          }
        }
      }
    }

    // output projection + residual: a = aPre @ wout; ar1 = x + a ; out = ar1 + m
    for (let t = 0; t < T; t++) {
      for (let o = 0; o < C; o++) {
        let s = 0;
        for (let i = 0; i < C; i++) s += lc.aPre[t * C + i] * wout[o * C + i];
        lc.a[t * C + o] = s;
        lc.ar1[t * C + o] = x[t * C + o] + s;
      }
    }

    // rmsnorm 2
    for (let t = 0; t < T; t++) {
      const off = t * C;
      rmsnorm(lc.ar1.subarray(off, off + C), mnorm, eps, lc.xn2.subarray(off, off + C) as Float64Array, {
        x: lc.mn.x.subarray(off, off + C) as Float64Array,
        r: lc.mn.r.subarray(off, off + C) as Float64Array,
      });
    }

    // MLP: pre = xn2 @ w1; h = gelu(pre); m = h @ w2; out = ar1 + m
    for (let t = 0; t < T; t++) {
      for (let o = 0; o < MLP; o++) {
        let s = 0;
        for (let i = 0; i < C; i++) s += lc.xn2[t * C + i] * w1[o * C + i];
        lc.pre[t * MLP + o] = s;
        lc.hpost[t * MLP + o] = gelu(s);
      }
      for (let o = 0; o < C; o++) {
        let s = 0;
        for (let i = 0; i < MLP; i++) s += lc.hpost[t * MLP + i] * w2[o * MLP + i];
        lc.m[t * C + o] = s;
        lc.out[t * C + o] = lc.ar1[t * C + o] + s;
      }
    }
  }

  // ================= BACKWARD =================

  private backwardItem(tokens: Int32Array, targets: Int32Array, b: number, cache: ItemCache): void {
    const { v: V, c: C } = this;
    const T = this.cfg.blockSize;
    const B = targets.length / T;
    const bn = 1 / B; // mean over batch
    const wte = this.weights.get("wte")!;
    const dwte = this.grads.get("wte")!;
    const dwpe = this.grads.get("wpe")!;
    const fnorm = this.weights.get("fnorm")!;

    // dlogits from cross-entropy
    const dlogits = new Float64Array(V * T);
    for (let t = 0; t < T; t++) {
      const y = targets[b * T + t];
      let maxL = -Infinity;
      for (let j = 0; j < V; j++) if (cache.logits[t * V + j] > maxL) maxL = cache.logits[t * V + j];
      let sumExp = 0;
      for (let j = 0; j < V; j++) sumExp += Math.exp(cache.logits[t * V + j] - maxL);
      for (let j = 0; j < V; j++) {
        dlogits[t * V + j] = (Math.exp(cache.logits[t * V + j] - maxL) / sumExp - (j === y ? 1 : 0)) / T * bn;
      }
    }

    // head: dwte += dlogits^T @ ln ; dln = dlogits @ wte
    const dln = new Float64Array(C * T);
    for (let t = 0; t < T; t++) {
      for (let j = 0; j < V; j++) {
        const g = dlogits[t * V + j];
        if (g === 0) continue;
        for (let i = 0; i < C; i++) {
          dwte[j * C + i] += g * cache.ln[t * C + i];
          dln[t * C + i] += g * wte[j * C + i];
        }
      }
    }

    // final RMSNorm backward
    const dfnorm = this.grads.get("fnorm")!;
    const dx = new Float64Array(C * T);
    for (let t = 0; t < T; t++) {
      const off = t * C;
      const dnorm = dln.subarray(off, off + C);
      const gc = {
        x: cache.fln.x.subarray(off, off + C) as Float64Array,
        r: cache.fln.r.subarray(off, off + C) as Float64Array,
      };
      const out = new Float64Array(C);
      rmsnormBackward(dnorm, fnorm, dfnorm, gc, out);
      for (let i = 0; i < C; i++) dx[off + i] = out[i];
    }

    // blocks in reverse
    for (let l = this.cfg.nLayer - 1; l >= 0; l--) {
      this.backwardLayer(l, dx, cache.perLayer[l]);
    }

    // embedding grad
    for (let t = 0; t < T; t++) {
      const id = tokens[b * T + t];
      const base = t * C;
      for (let i = 0; i < C; i++) {
        dwte[id * C + i] += dx[base + i];
        dwpe[base + i] += dx[base + i];
      }
    }
  }

  private backwardLayer(l: number, dOut: Float64Array, lc: LayerCache): void {
    const { c: C, nHead: H, headDim: HD, mlpDim: MLP } = this;
    const T = this.cfg.blockSize;
    const w1 = this.weights.get(`b${l}.w1`)!;
    const dw1 = this.grads.get(`b${l}.w1`)!;
    const w2 = this.weights.get(`b${l}.w2`)!;
    const dw2 = this.grads.get(`b${l}.w2`)!;
    const wqkv = this.weights.get(`b${l}.wqkv`)!;
    const dwqkv = this.grads.get(`b${l}.wqkv`)!;
    const wout = this.weights.get(`b${l}.wout`)!;
    const dwout = this.grads.get(`b${l}.wout`)!;
    const anorm = this.weights.get(`b${l}.anorm`)!;
    const danorm = this.grads.get(`b${l}.anorm`)!;
    const mnorm = this.weights.get(`b${l}.mnorm`)!;
    const dmnorm = this.grads.get(`b${l}.mnorm`)!;
    const scale = 1 / Math.sqrt(HD);

    // ---- MLP backward ----
    // out = ar1 + m  =>  dAr1 = dM = dOut
    const dM = new Float64Array(C * T);
    const dAr1 = new Float64Array(C * T);
    for (let t = 0; t < T; t++) {
      for (let o = 0; o < C; o++) {
        const g = dOut[t * C + o];
        dM[t * C + o] = g;
        dAr1[t * C + o] = g;
      }
    }
    // m = hpost @ w2  =>  dw2 += hpost^T @ dM ; dh = dM @ w2^T
    const dh = new Float64Array(MLP * T);
    for (let t = 0; t < T; t++) {
      for (let o = 0; o < C; o++) {
        const g = dM[t * C + o];
        for (let i = 0; i < MLP; i++) {
          dw2[o * MLP + i] += lc.hpost[t * MLP + i] * g;
          dh[t * MLP + i] += g * w2[o * MLP + i];
        }
      }
    }
    // hpost = gelu(pre) => dpre ; pre = xn2 @ w1 => dw1 += xn2^T @ dpre ; dxn2 = dpre @ w1^T
    const dPre = new Float64Array(MLP * T);
    const dxn2 = new Float64Array(C * T);
    for (let t = 0; t < T; t++) {
      for (let o = 0; o < MLP; o++) {
        const g = dh[t * MLP + o] * geluDerivative(lc.pre[t * MLP + o]);
        dPre[t * MLP + o] = g;
        for (let i = 0; i < C; i++) {
          dw1[o * C + i] += lc.xn2[t * C + i] * g;
          dxn2[t * C + i] += g * w1[o * C + i];
        }
      }
    }
    // rmsnorm2 backward
    const dAr1Norm = new Float64Array(C * T);
    for (let t = 0; t < T; t++) {
      const off = t * C;
      const gc = {
        x: lc.mn.x.subarray(off, off + C) as Float64Array,
        r: lc.mn.r.subarray(off, off + C) as Float64Array,
      };
      const out = new Float64Array(C);
      rmsnormBackward(dxn2.subarray(off, off + C), mnorm, dmnorm, gc, out);
      for (let i = 0; i < C; i++) dAr1Norm[off + i] = out[i];
    }
    // combine
    for (let i = 0; i < C * T; i++) dAr1[i] += dAr1Norm[i];

    // ---- attention backward ----
    // ar1 = x + a ; a = aPre @ wout
    const dAPre = new Float64Array(C * T);
    const dxAttn = new Float64Array(C * T);
    for (let t = 0; t < T; t++) {
      for (let o = 0; o < C; o++) {
        const g = dAr1[t * C + o];
        for (let i = 0; i < C; i++) {
          dwout[o * C + i] += lc.aPre[t * C + i] * g;
          dAPre[t * C + i] += g * wout[o * C + i];
        }
        dxAttn[t * C + o] += g;
      }
    }
    // per-head: dv, dpre-softmax scores
    const dQ = new Float64Array(H * T * HD);
    const dK = new Float64Array(H * T * HD);
    const dV = new Float64Array(H * T * HD);
    const dScores = new Float64Array(H * T * T);
    for (let h = 0; h < H; h++) {
      const hb = h * (T * HD);
      const colBase = h * HD;
      // dScores (un-normalized) = sum_d dAPre * v ; dv = sum_t probs * dAPre
      for (let t = 0; t < T; t++) {
        for (let t2 = 0; t2 <= t; t2++) {
          let s = 0;
          for (let d = 0; d < HD; d++) {
            const g = dAPre[t * C + colBase + d];
            s += g * lc.vh[hb + t2 * HD + d];
            dV[hb + t2 * HD + d] += g * lc.probs[h * T * T + t * T + t2];
          }
          dScores[h * T * T + t * T + t2] = s;
        }
      }
      // softmax backward
      for (let t = 0; t < T; t++) {
        let dotP = 0;
        for (let t2 = 0; t2 <= t; t2++) dotP += lc.probs[h * T * T + t * T + t2] * dScores[h * T * T + t * T + t2];
        for (let t2 = 0; t2 <= t; t2++) {
          dScores[h * T * T + t * T + t2] = lc.probs[h * T * T + t * T + t2] * (dScores[h * T * T + t * T + t2] - dotP);
        }
      }
      // dq, dk
      for (let t = 0; t < T; t++) {
        for (let t2 = 0; t2 <= t; t2++) {
          const g = dScores[h * T * T + t * T + t2];
          if (g === 0) continue;
          for (let d = 0; d < HD; d++) {
            dQ[hb + t * HD + d] += g * lc.kh[hb + t2 * HD + d] * scale;
            dK[hb + t2 * HD + d] += g * lc.qh[hb + t * HD + d] * scale;
          }
        }
      }
    }

    // wqkv grads + dn1
    const dxn1 = new Float64Array(C * T);
    for (let t = 0; t < T; t++) {
      for (let o = 0; o < 3 * C; o++) {
        const src = o < C ? dQ : o < 2 * C ? dK : dV;
        const col = o % C;
        const h = Math.floor(col / HD);
        const d = col % HD;
        const g = src[h * (T * HD) + t * HD + d];
        for (let i = 0; i < C; i++) {
          dwqkv[o * C + i] += lc.xn1[t * C + i] * g;
          dxn1[t * C + i] += g * wqkv[o * C + i];
        }
      }
    }

    // rmsnorm1 backward + residual: dx = dxAttn + dnorm1
    const dx = new Float64Array(C * T);
    for (let t = 0; t < T; t++) {
      const off = t * C;
      const gc = {
        x: lc.an.x.subarray(off, off + C) as Float64Array,
        r: lc.an.r.subarray(off, off + C) as Float64Array,
      };
      const out = new Float64Array(C);
      rmsnormBackward(dxn1.subarray(off, off + C), anorm, danorm, gc, out);
      for (let i = 0; i < C; i++) dx[off + i] = dxAttn[off + i] + out[i];
    }
    for (let i = 0; i < C * T; i++) dOut[i] = dx[i];
  }
}

export interface RMSNormRes {
  x: Float64Array;
  r: Float64Array;
}

export interface LayerCache {
  xn1: Float64Array;
  an: RMSNormRes;
  ar1: Float64Array;
  qh: Float64Array;
  kh: Float64Array;
  vh: Float64Array;
  probs: Float64Array;
  aPre: Float64Array;
  a: Float64Array;
  xn2: Float64Array;
  mn: RMSNormRes;
  pre: Float64Array;
  hpost: Float64Array;
  m: Float64Array;
  out: Float64Array;
}

export interface ItemCache {
  x: Float64Array;
  loss: number;
  perLayer: LayerCache[];
  ln: Float64Array;
  fln: RMSNormRes;
  logits: Float64Array;
}

/** AdamW optimizer with per-parameter moment buffers. */
export class AdamW {
  private readonly m: Map<string, Float64Array> = new Map();
  private readonly v: Map<string, Float64Array> = new Map();
  private step = 0;

  constructor(private readonly model: GPT) {
    for (const name of model.order) {
      this.m.set(name, new Float64Array(model.weights.get(name)!.length));
      this.v.set(name, new Float64Array(model.weights.get(name)!.length));
    }
  }

  /** Zero the moment buffers (e.g. when reusing an optimizer on a reset model). */
  reset(): void {
    for (const name of this.model.order) {
      this.m.get(name)!.fill(0);
      this.v.get(name)!.fill(0);
    }
  }

  /** Apply one update. Returns the pre-clip gradient norm. */
  stepOnce(cfg: TrainingConfig): number {
    const model = this.model;
    const lr = cfg.learningRate;
    let gradNorm = 0;
    for (const name of model.order) {
      const g = model.grads.get(name)!;
      for (let i = 0; i < g.length; i++) gradNorm += g[i] * g[i];
    }
    gradNorm = Math.sqrt(gradNorm);
    const clip = cfg.gradClip > 0 && gradNorm > cfg.gradClip ? cfg.gradClip / gradNorm : 1;
    this.step += 1;
    const b1t = 1 - Math.pow(cfg.beta1, this.step);
    const b2t = 1 - Math.pow(cfg.beta2, this.step);
    for (const name of model.order) {
      const w = model.weights.get(name)!;
      const g = model.grads.get(name)!;
      const m = this.m.get(name)!;
      const v = this.v.get(name)!;
      for (let i = 0; i < w.length; i++) {
        const gv = g[i] * clip;
        m[i] = cfg.beta1 * m[i] + (1 - cfg.beta1) * gv;
        v[i] = cfg.beta2 * v[i] + (1 - cfg.beta2) * gv * gv;
        const mhat = m[i] / b1t;
        const vhat = v[i] / b2t;
        w[i] -= lr * (mhat / (Math.sqrt(vhat) + cfg.eps) + cfg.weightDecay * w[i]);
      }
    }
    return gradNorm;
  }
}

function gauss(): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}