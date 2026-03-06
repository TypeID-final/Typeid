const avg = (a) => a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
const safeNum = (v, fallback = 0) => (Number.isFinite(v) ? v : fallback);
const std = (a) => {
  if (a.length < 2) return 0;
  const m = avg(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length);
};
const cv = (a) => { const m = avg(a); return m > 0 ? std(a) / m : 0; };

function entropy(lst) {
  if (lst.length < 2) return 0;
  const mn = Math.min(...lst), mx = Math.max(...lst);
  if (mx === mn) return 0;
  const bins = 10, w = (mx - mn) / bins;
  const hist = new Array(bins).fill(0);
  lst.forEach(v => { const b = Math.min(bins - 1, Math.floor((v - mn) / w)); hist[b]++; });
  return -hist.reduce((s, c) => { if (!c) return s; const p = c / lst.length; return s + p * Math.log2(p); }, 0);
}

const DIGRAPH_MAP = {
  "th": "dg_th", "he": "dg_he", "qu": "dg_qu", "br": "dg_br", "ow": "dg_ow",
  "fo": "dg_fo", "ju": "dg_ju", "ov": "dg_ov", "er": "dg_er", "la": "dg_la",
};
const TRIGRAPH_MAP = {
  "the": "tg_the", "bro": "tg_bro", "own": "tg_own", "ove": "tg_ove", "ver": "tg_ver",
};
const KEY_DWELL_KEYS = ["e", "o", "t", "h", "r", "u", "space"];

export const freshState = () => ({
  keyDownTimes: {},
  dwellTimes: [],
  keyDwells: {},
  flightTimes: [],
  allIntervals: [],
  lastKeyUpTime: null,
  keyCount: 0,
  backspaceCount: 0,
  startTime: null,
  dgTimes: {},
  tgTimes: {},
  keySeq: [],
});

export const handleKeyDownEvent = (e, ksState) => {
  const now = performance.now();
  if (!ksState.startTime) ksState.startTime = now;
  ksState.keyDownTimes[e.code] = now;

  if (ksState.lastKeyUpTime !== null) {
    const flight = now - ksState.lastKeyUpTime;
    ksState.flightTimes.push(flight);
    ksState.allIntervals.push(flight);
  }

  if (e.key === "Backspace") ksState.backspaceCount++;
  ksState.keyCount++;

  const k = e.key === " " ? "space" : e.key.toLowerCase();
  ksState.keySeq.push({ key: k, time: now });

  if (ksState.keySeq.length >= 2) {
    const p = ksState.keySeq[ksState.keySeq.length - 2];
    const c = ksState.keySeq[ksState.keySeq.length - 1];
    const pair = p.key + c.key;
    const feat = DIGRAPH_MAP[pair];
    if (feat) {
      if (!ksState.dgTimes[feat]) ksState.dgTimes[feat] = [];
      const gap = c.time - p.time;
      if (gap >= 20 && gap <= 1500) ksState.dgTimes[feat].push(gap);
    }
    if (c.key === "space") {
      if (!ksState.dgTimes["dg_sp"]) ksState.dgTimes["dg_sp"] = [];
      const gap = c.time - p.time;
      if (gap >= 20 && gap <= 1500) ksState.dgTimes["dg_sp"].push(gap);
    }
  }

  if (ksState.keySeq.length >= 3) {
    const k1 = ksState.keySeq[ksState.keySeq.length - 3];
    const k2 = ksState.keySeq[ksState.keySeq.length - 2];
    const k3 = ksState.keySeq[ksState.keySeq.length - 1];
    const tri = k1.key + k2.key + k3.key;
    const feat = TRIGRAPH_MAP[tri];
    if (feat) {
      if (!ksState.tgTimes[feat]) ksState.tgTimes[feat] = [];
      const span = k3.time - k1.time;
      if (span >= 30 && span <= 3000) ksState.tgTimes[feat].push(span);
    }
  }
};

export const handleKeyUpEvent = (e, ksState) => {
  const now = performance.now();
  if (ksState.keyDownTimes[e.code] !== undefined) {
    const dwell = now - ksState.keyDownTimes[e.code];
    if (dwell >= 20 && dwell <= 800) {
      ksState.dwellTimes.push(dwell);
      const k = e.key === " " ? "space" : e.key.toLowerCase();
      if (KEY_DWELL_KEYS.includes(k)) {
        if (!ksState.keyDwells[k]) ksState.keyDwells[k] = [];
        ksState.keyDwells[k].push(dwell);
      }
    }
    delete ksState.keyDownTimes[e.code];
  }
  ksState.lastKeyUpTime = now;
};

export const buildFeatures = (ksState, inputVal) => {
  const elapsedSec = ksState.startTime ? (performance.now() - ksState.startTime) / 1000 : 0;
  const words = inputVal.trim().split(/\s+/).filter(Boolean).length;
  const dw = ksState.dwellTimes, fl = ksState.flightTimes;
  const dwell_mean = avg(dw), flight_mean = avg(fl);
  const dg = (feat) => avg(ksState.dgTimes[feat] || []);
  const tg = (feat) => avg(ksState.tgTimes[feat] || []);
  const kd = (key) => avg(ksState.keyDwells[key] || []);

  return {
    dwell_mean: +(dwell_mean || 120).toFixed(2),
    dwell_std: +(std(dw) || 25).toFixed(2),
    dwell_cv: +(cv(dw)).toFixed(4),
    flight_mean: +(flight_mean || 150).toFixed(2),
    flight_std: +(std(fl) || 30).toFixed(2),
    flight_cv: +(cv(fl)).toFixed(4),
    timing_entropy: +(entropy([...dw, ...fl])).toFixed(4),
    total_duration: +(elapsedSec * 1000 || 5000).toFixed(2),
    wpm: +(elapsedSec > 0 ? (words / elapsedSec * 60) : 90).toFixed(2),
    dg_th: +dg("dg_th").toFixed(2), dg_he: +dg("dg_he").toFixed(2),
    dg_qu: +dg("dg_qu").toFixed(2), dg_br: +dg("dg_br").toFixed(2),
    dg_ow: +dg("dg_ow").toFixed(2), dg_fo: +dg("dg_fo").toFixed(2),
    dg_ju: +dg("dg_ju").toFixed(2), dg_ov: +dg("dg_ov").toFixed(2),
    dg_er: +dg("dg_er").toFixed(2), dg_la: +dg("dg_la").toFixed(2),
    dg_sp: +dg("dg_sp").toFixed(2),
    tg_the: +tg("tg_the").toFixed(2), tg_bro: +tg("tg_bro").toFixed(2),
    tg_own: +tg("tg_own").toFixed(2), tg_ove: +tg("tg_ove").toFixed(2),
    tg_ver: +tg("tg_ver").toFixed(2),
    kd_e: +kd("e").toFixed(2), kd_o: +kd("o").toFixed(2),
    kd_t: +kd("t").toFixed(2), kd_h: +kd("h").toFixed(2),
    kd_r: +kd("r").toFixed(2), kd_u: +kd("u").toFixed(2),
    kd_space: +kd("space").toFixed(2),
    backspace_rate: ksState.keyCount > 0 ? +(ksState.backspaceCount / ksState.keyCount).toFixed(4) : 0,
  };
};
