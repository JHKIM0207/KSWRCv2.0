// Run periodically (GitHub Actions) to accumulate NOAA RTSW solar-wind history
// into daily JSONL files, since the live NOAA feed only exposes a short rolling window.
import fs from 'node:fs/promises';
import path from 'node:path';

const WIND_URL = 'https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json';
const MAG_URL = 'https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json';
const DIR = path.join(process.cwd(), 'data', 'wind-history');
const KEEP_DAYS = 5;

async function fetchJson(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(`${url} -> HTTP ${r.status}`);
  return r.json();
}

function finiteValue(value) {
  if (value === null || value === undefined || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

// NOAA's replacement RTSW products expose all spacecraft and use `active` as
// an operational-spacecraft flag, not as a measurement-quality flag. Prefer
// the active spacecraft, but retain a valid inactive source when no active row
// exists for that timestamp (for example during a forecaster source switch).
function preferredRows(rows, fields) {
  const byTime = new Map();
  for (const row of rows || []) {
    if (!row?.time_tag) continue;
    const t = Date.parse(/Z$|[+-]\d\d:?\d\d$/.test(row.time_tag) ? row.time_tag : row.time_tag + 'Z');
    if (!Number.isFinite(t)) continue;
    const validCount = fields.reduce((count, field) => count + (finiteValue(row[field]) !== null ? 1 : 0), 0);
    if (!validCount) continue;
    const candidate = { t, r: row, score: (row.active === true ? 100 : 0) + validCount };
    const previous = byTime.get(t);
    if (!previous || candidate.score > previous.score) byTime.set(t, candidate);
  }
  return [...byTime.values()].sort((a, b) => a.t - b.t);
}

function dayKey(t) {
  return new Date(t).toISOString().slice(0, 10);
}

async function loadDayFile(day) {
  try {
    const text = await fs.readFile(path.join(DIR, `${day}.jsonl`), 'utf8');
    const map = new Map();
    for (const line of text.split('\n')) {
      if (!line.trim()) continue;
      const p = JSON.parse(line);
      map.set(p.t, p);
    }
    return map;
  } catch {
    return new Map();
  }
}

async function saveDayFile(day, map) {
  const points = [...map.values()].sort((a, b) => a.t - b.t);
  const text = points.map(p => JSON.stringify(p)).join('\n') + '\n';
  await fs.writeFile(path.join(DIR, `${day}.jsonl`), text, 'utf8');
}

async function main() {
  await fs.mkdir(DIR, { recursive: true });
  const [windRaw, magRaw] = await Promise.all([
    fetchJson(WIND_URL).catch(e => { console.warn('wind fetch failed', e); return []; }),
    fetchJson(MAG_URL).catch(e => { console.warn('mag fetch failed', e); return []; })
  ]);

  const windByT = new Map(preferredRows(windRaw, ['proton_speed', 'proton_density']).map(({ t, r }) => [t, r]));
  const magByT = new Map(preferredRows(magRaw, ['bt', 'bx_gsm', 'by_gsm', 'bz_gsm']).map(({ t, r }) => [t, r]));
  const allT = new Set([...windByT.keys(), ...magByT.keys()]);

  const byDay = new Map();
  for (const t of allT) {
    const w = windByT.get(t), m = magByT.get(t);
    const speed = finiteValue(w?.proton_speed), density = finiteValue(w?.proton_density);
    const bt = finiteValue(m?.bt), bx = finiteValue(m?.bx_gsm), by = finiteValue(m?.by_gsm), bz = finiteValue(m?.bz_gsm);
    const point = {
      t,
      speed: speed !== null && speed >= 100 && speed <= 3000 ? speed : null,
      density: density !== null && density >= 0.05 && density <= 500 ? density : null,
      bt: bt !== null && bt >= 0 && bt <= 300 ? bt : null,
      bx: bx !== null && Math.abs(bx) <= 300 ? bx : null,
      by: by !== null && Math.abs(by) <= 300 ? by : null,
      bz: bz !== null && Math.abs(bz) <= 300 ? bz : null
    };
    const day = dayKey(t);
    if (!byDay.has(day)) byDay.set(day, []);
    byDay.get(day).push(point);
  }

  for (const [day, points] of byDay) {
    const map = await loadDayFile(day);
    for (const p of points) map.set(p.t, p);
    await saveDayFile(day, map);
  }

  // Prune files older than KEEP_DAYS
  const cutoff = Date.now() - KEEP_DAYS * 86400000;
  const files = await fs.readdir(DIR).catch(() => []);
  for (const f of files) {
    const m = f.match(/^(\d{4}-\d{2}-\d{2})\.jsonl$/);
    if (!m) continue;
    if (Date.parse(m[1] + 'T23:59:59Z') < cutoff) await fs.unlink(path.join(DIR, f)).catch(() => {});
  }

  // Manifest of available day files, for the page to know what to fetch.
  const manifest = (await fs.readdir(DIR).catch(() => []))
    .filter(f => /^\d{4}-\d{2}-\d{2}\.jsonl$/.test(f))
    .map(f => f.replace('.jsonl', ''))
    .sort();
  await fs.writeFile(path.join(DIR, 'manifest.json'), JSON.stringify(manifest), 'utf8');

  const activeWind = windRaw.filter(row => row?.active === true).length;
  const selectedInactiveWind = [...windByT.values()].filter(row => row?.active !== true).length;
  console.log(`Updated ${byDay.size} day file(s): ${[...byDay.keys()].join(', ')}`);
  console.log(`Wind rows: ${windByT.size} selected (${activeWind} active source rows, ${selectedInactiveWind} inactive fallbacks)`);
}

main().catch(e => { console.error(e); process.exit(1); });
