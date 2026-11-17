// gta6-twitch-ccu-forecast-01: forecast GTA VI's launch-day Twitch viewers, hour by hour,
// from the case's hour-1 print and the release-day curves of past big launches.
//
// Model (fitted at run time from the case file's own `comparables`, never from constants
// tuned on any one launch): for each hour h = 2..24, the log of that hour's viewers is
// regressed on the log of hour 1's print, ln(v_h) = a_h + b_h * ln(p1), with a Gaussian
// kernel over ln(p1) (bandwidth 1.5 log units) so launches whose opening hour looked like
// this launch's count most. The spread is the kernel-weighted residual standard deviation
// (shrunk by 0.75), so the five quantiles are exp(mu_h + z_tau * s_h).
//
// Why this shape: across past launches the ratio viewers(h)/viewers(1) depends strongly on
// how big hour 1 already was (correlation about -0.9 by the evening: a launch whose first
// hour is already huge, because most of the world unlocked before UTC midnight, barely
// grows past it; a launch whose first hour is tiny, because its unlock lands later in the
// UTC day, explodes tenfold). A flat persistence forecast misses that; a single median
// ratio misses it too. The kernel-weighted slope captures it, and the residual spread
// keeps the bands honest (about 0.83 coverage in leave-one-out checks over the
// comparables, 0.95 on the practice cases).
//
// Falls back gracefully, hour by hour, so a case is always valid: kernel fit -> unweighted
// fit on all launches with that hour -> median log-ratio -> flat persistence with a
// widening spread. No network, no randomness, no state: two runs write identical bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Gaussian kernel bandwidth over ln(hour-1 viewers): launches this close in log-opening
 *  size dominate a target's fit. */
const KERNEL_BW = 1.5;
/** Residual-spread shrink: the raw residual sd is slightly wide, so the 90% band is
 *  calibrated a touch narrower. Calibrated 0.75 on the practice cases (mean pinball loss
 *  0.1855 -> 0.1773, every case better, coverage 0.989 -> 0.946) and on leave-one-out
 *  over the 20 comparables with a big opening hour (0.2056 -> 0.2033); narrower (0.6)
 *  scores a hair better on practice but degrades leave-one-out, so 0.75 is the honest
 *  middle, not the practice minimum. */
const SPREAD_SHRINK = 0.75;
/** Guards on the log spread actually used. */
const SD_FLOOR = 0.05;
const SD_CAP = 1.5;
/** The persistence fallback's spread, in log units per square-root hour (house baseline). */
const FALLBACK_SIGMA = 0.15;

interface HourlyPoint {
  from: string;
  viewers: number;
  channels?: number | null;
}

interface Comparable {
  name: string;
  day_start: string;
  hourly: HourlyPoint[];
}

interface CaseFile {
  id: string;
  levels: number[];
  known: Array<{ hour: number; viewers: number; channels?: number | null }>;
  forecast: Array<{ hour: number }>;
  comparables?: Comparable[];
}

/** One past launch, reduced to what the fit needs: its log opening print and its 24
 *  release-day hourly viewer counts (null where the source skipped an hour). */
interface Launch {
  logP1: number;
  day: Array<number | null>;
}

/** Acklam's rational approximation to the standard normal's quantile (inverse CDF),
 *  accurate to ~1e-9: the z-score at each scored level. */
const NORM_A = [-3.969683028665376e1, 2.209460984245205e2, -2.759285984469534e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239e0];
const NORM_B = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
const NORM_C = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0, -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0];
const NORM_D = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0, 3.754408661907416e0];
const NORM_P_LOW = 0.02425;

function normInv(p: number): number {
  if (!(p > 0 && p < 1)) throw new Error(`level ${p} is outside (0, 1)`);
  let q: number;
  if (p < NORM_P_LOW) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((NORM_C[0] * q + NORM_C[1]) * q + NORM_C[2]) * q + NORM_C[3]) * q + NORM_C[4]) * q + NORM_C[5]) /
      ((((NORM_D[0] * q + NORM_D[1]) * q + NORM_D[2]) * q + NORM_D[3]) * q + 1);
  }
  if (p <= 1 - NORM_P_LOW) {
    q = p - 0.5;
    const r = q * q;
    return (((((NORM_A[0] * r + NORM_A[1]) * r + NORM_A[2]) * r + NORM_A[3]) * r + NORM_A[4]) * r + NORM_A[5]) * q /
      (((((NORM_B[0] * r + NORM_B[1]) * r + NORM_B[2]) * r + NORM_B[3]) * r + NORM_B[4]) * r + 1);
  }
  q = Math.sqrt(-2 * Math.log(1 - p));
  return -(((((NORM_C[0] * q + NORM_C[1]) * q + NORM_C[2]) * q + NORM_C[3]) * q + NORM_C[4]) * q + NORM_C[5]) /
    ((((NORM_D[0] * q + NORM_D[1]) * q + NORM_D[2]) * q + NORM_D[3]) * q + 1);
}

/** x rounded to 2 decimals (half up), as the baseline rounds its quantiles. */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** Weighted least squares for y = a + b*x with weights w (2x2 normal equations, tiny
 *  ridge for stability). Returns [a, b, residualSd] or null when the fit is degenerate. */
function weightedOls(
  xs: number[],
  ys: number[],
  ws: number[],
): [number, number, number] | null {
  const n = xs.length;
  if (n < 3) return null;
  const ridge = 1e-9;
  let sw = 0;
  let swx = 0;
  let swy = 0;
  let swxx = 0;
  let swxy = 0;
  for (let i = 0; i < n; i++) {
    const w = ws[i];
    sw += w;
    swx += w * xs[i];
    swy += w * ys[i];
    swxx += w * xs[i] * xs[i];
    swxy += w * xs[i] * ys[i];
  }
  // Effective sample size and weighted variance of x: a fit that leans on one launch
  // (or on launches all the same size) is not a fit.
  const effN = (sw * sw) / ws.reduce((acc, w) => acc + w * w, 0);
  const meanX = swx / sw;
  const varX = swxx / sw - meanX * meanX;
  if (!(effN >= 4) || !(varX > 1e-6) || !(sw > 0)) return null;
  const det = (sw + ridge) * (swxx + ridge) - swx * swx;
  if (!(Math.abs(det) > 1e-12)) return null;
  const a = ((swxx + ridge) * swy - swx * swxy) / det;
  const b = ((sw + ridge) * swxy - swx * swy) / det;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (a + b * xs[i]);
    ss += ws[i] * r * r;
  }
  const dof = Math.max(effN - 2, 1);
  const sd = Math.sqrt(ss / sw / dof * effN);
  return [a, b, sd];
}

/** The median of a list of numbers (assumes non-empty). */
function medianOf(values: number[]): number {
  const sorted = [...values].sort((x, y) => x - y);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** Fit hour h (2..24) for a target whose log opening print is logP1, from the launches.
 *  Returns { mu, sd } in log-viewer units. */
function fitHour(launches: Launch[], logP1: number, h: number): { mu: number; sd: number } {
  const rows = launches
    .map((launch) => ({ x: launch.logP1, y: launch.day[h - 1] === null ? null : Math.log(launch.day[h - 1] as number) }))
    .filter((row): row is { x: number; y: number } => row.y !== null && Number.isFinite(row.y));
  const xs = rows.map((r) => r.x);
  const ys = rows.map((r) => r.y);

  // Primary: Gaussian-kernel-weighted fit around this launch's opening size.
  const kernelWs = xs.map((x) => Math.exp(-0.5 * ((x - logP1) / KERNEL_BW) ** 2));
  const kernel = weightedOls(xs, ys, kernelWs);
  if (kernel !== null) {
    const [a, b, sd] = kernel;
    return { mu: a + b * logP1, sd: Math.min(Math.max(sd * SPREAD_SHRINK, SD_FLOOR), SD_CAP) };
  }

  // Fallback: unweighted fit on every launch that has this hour.
  const plain = weightedOls(xs, ys, xs.map(() => 1));
  if (plain !== null) {
    const [a, b, sd] = plain;
    return { mu: a + b * logP1, sd: Math.min(Math.max(sd * SPREAD_SHRINK, SD_FLOOR), SD_CAP) };
  }

  // Fallback: the median log-ratio to hour 1, with the cross-launch spread around it.
  const ratios = rows.map((r) => r.y - r.x);
  if (ratios.length >= 2) {
    const med = medianOf(ratios);
    const mean = ratios.reduce((acc, v) => acc + v, 0) / ratios.length;
    const sd = Math.sqrt(ratios.reduce((acc, v) => acc + (v - mean) ** 2, 0) / (ratios.length - 1));
    return { mu: logP1 + med, sd: Math.min(Math.max(sd * SPREAD_SHRINK, SD_FLOOR), SD_CAP) };
  }

  // Last resort: flat persistence with the baseline's widening spread.
  return { mu: logP1, sd: FALLBACK_SIGMA * Math.sqrt(h - 1) };
}

/** A comparable's release-day curve: 24 hourly counts from its day_start, null where the
 *  source skipped an hour. Returns null when the launch has no usable opening print
 *  (this also drops non-launches such as a category's pre-release history). */
function launchOf(comp: Comparable): Launch | null {
  const byFrom = new Map<string, number>();
  for (const p of comp.hourly ?? []) {
    if (typeof p.viewers === 'number' && Number.isFinite(p.viewers)) byFrom.set(p.from, p.viewers);
  }
  const startMs = Date.parse(comp.day_start);
  if (!Number.isFinite(startMs)) return null;
  const day: Array<number | null> = [];
  for (let h = 0; h < 24; h++) {
    const at = new Date(startMs + h * 3600_000).toISOString().replace('.000Z', 'Z');
    day.push(byFrom.has(at) ? (byFrom.get(at) as number) : null);
  }
  if (day[0] === null || day[0] <= 0) return null;
  return { logP1: Math.log(day[0]), day };
}

/** The five quantiles of hour h for one case, from the fitted log-mean and log-spread. */
function hourQuantiles(mu: number, sd: number, zs: number[]): number[] {
  const qs = zs.map((z) => round2(Math.exp(mu + z * sd)));
  // Enforce the format's rules: above 0, at most 1e9, non-decreasing.
  let prev = 0;
  return qs.map((q) => {
    const clamped = Math.min(Math.max(q, 0.01), 1_000_000_000);
    const fixed = Math.max(clamped, prev);
    prev = fixed;
    return fixed;
  });
}

/** Forecast one case file. */
function forecastCase(c: CaseFile): { hours: Array<{ hour: number; q: number[] }> } {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`case ${c.id} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const zs = LEVELS.map((t) => normInv(t));
  const first = (c.known ?? []).find((k) => k.hour === 1);
  const p1 = Math.max(1, first !== undefined && Number.isFinite(first.viewers) ? first.viewers : 1);
  const logP1 = Math.log(p1);
  const launches: Launch[] = [];
  for (const comp of c.comparables ?? []) {
    const launch = launchOf(comp);
    if (launch !== null) launches.push(launch);
  }
  const hours = c.forecast.map(({ hour }) => {
    const { mu, sd } = launches.length > 0
      ? fitHour(launches, logP1, hour)
      : { mu: logP1, sd: FALLBACK_SIGMA * Math.sqrt(hour - 1) };
    return { hour, q: hourQuantiles(mu, sd, zs) };
  });
  return { hours };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  const out = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}
