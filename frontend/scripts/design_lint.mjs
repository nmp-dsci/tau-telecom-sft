#!/usr/bin/env node
/**
 * The house UI's never-do list as a check (after finance-rag-sft's, itself after vidore-v3-loop's).
 *
 * Every colour in this app comes from ../.lavish/house.css (tokens) and ../.lavish/project.css
 * (the project's one number, --hue). So, anywhere under src/:
 *   - no hex colour literal (#abc, #aabbcc, #aabbccdd);
 *   - no rgb()/rgba()/hsl()/hsla()/oklch()/oklab()/lab()/lch()/hwb()/color() literal;
 *   - no named colour as a colour value (color: red);
 *   - no declaration of --hue (the registry owns it, via project.css);
 *   - no font size under 12.8px (--t-2 = 0.8rem), in CSS, JSX style objects or SVG attributes.
 * No dependencies; runs in CI from the frontend job.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const srcDir = join(root, 'src');
const MIN_PX = 12.8;
const MIN_REM = 0.8;

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (/\.(tsx?|css|mjs|js|html|svg)$/.test(name)) out.push(p);
  }
  return out;
}

const HEX = /(?<![&\w])#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3,4})\b/g;
const COLOUR_FN = /\b(rgba?|hsla?|oklch|oklab|lab|lch|hwb|color)\(/g;
const NAMED = new Set([
  'black', 'white', 'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'gray', 'grey', 'brown',
  'crimson', 'maroon', 'navy', 'teal', 'olive', 'lime', 'aqua', 'fuchsia', 'silver', 'gold', 'indigo', 'violet',
  'cyan', 'magenta', 'salmon', 'tomato', 'coral', 'khaki', 'beige', 'ivory', 'tan', 'firebrick', 'darkred',
]);
const COLOUR_PROP = /(?:^|[;{\s"'])(color|background|background-color|border-color|outline-color|fill|stroke)\s*:\s*['"]?([a-zA-Z]+)['"]?\s*[;}'",]/g;
const HUE_DECL = /--hue\s*:/g;
const CSS_SIZE = /font-size\s*:\s*(\d*\.?\d+)(px|rem|em)\b/g;
const CSS_FONT = /(?:^|[;{\s])font\s*:[^;}]*?\s(\d*\.?\d+)(px|rem|em)\b/g;
const JSX_SIZE = /fontSize\s*:\s*['"]?(\d*\.?\d+)(px|rem|em)?['"]?/g;
const SVG_SIZE = /font-size=["'{]\s*(\d*\.?\d+)(px|rem|em)?/g;

const tooSmall = (n, unit) => (unit === 'rem' || unit === 'em' ? n < MIN_REM : n < MIN_PX);
const line = (text, idx) => text.slice(0, idx).split('\n').length;

const problems = [];
for (const file of walk(srcDir)) {
  const rel = relative(root, file);
  const text = readFileSync(file, 'utf8');
  const at = (m) => `${rel}:${line(text, m.index)}`;
  for (const m of text.matchAll(HEX)) problems.push(`${at(m)}: hex colour ${m[0]} — use a token`);
  for (const m of text.matchAll(COLOUR_FN)) problems.push(`${at(m)}: colour literal ${m[1]}() — use a token`);
  for (const m of text.matchAll(COLOUR_PROP)) {
    if (NAMED.has(m[2].toLowerCase())) problems.push(`${at(m)}: named colour ${m[2]} — use a token`);
  }
  for (const m of text.matchAll(HUE_DECL)) problems.push(`${at(m)}: --hue is set by project.css only`);
  for (const re of [CSS_SIZE, CSS_FONT, JSX_SIZE, SVG_SIZE]) {
    for (const m of text.matchAll(re)) {
      const n = Number.parseFloat(m[1]);
      const unit = m[2] ?? 'px';
      if (tooSmall(n, unit)) problems.push(`${at(m)}: font size ${m[1]}${unit} is under --t-2 (12.8px)`);
    }
  }
}
for (const p of problems) console.log(`  ✗ ${p}`);
if (problems.length) {
  console.log(`design lint failed: ${problems.length}`);
  process.exit(1);
}
console.log('design lint: clean');
