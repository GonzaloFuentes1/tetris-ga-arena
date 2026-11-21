// Evaluates Tetris GA submissions. Runs on a schedule in GitHub Actions.
//
// People submit from the Tetris GA Space: the page opens a discussion titled
// "[submission] <name>" on the leaderboard dataset, with the agent as JSON.
// Hugging Face sets the discussion author, so we know who sent it. This script
// plays each new agent on secret seeds, keeps the best entry per user in
// leaderboard.json, replies with the result and closes the discussion.
'use strict';
const T = require('./tetris.js');

const REPO = process.env.LB_REPO || 'GonzaloFuentes1/tetris-ga-leaderboard';
const TOKEN = process.env.HF_TOKEN;
const SEEDS = (process.env.EVAL_SEEDS || '').split(',').map(Number).filter(Number.isFinite).slice(0, 8);
const CAP = 20000; // pieces per side in a battle
// Level benchmark: 10 secret games per level, up to 10,000 pieces each,
// measured in lines cleared per game. The games come from the secret seeds.
const LEVEL_CAP = 10000;
const LEVEL_SEEDS = (() => { const r = T.rng((SEEDS[0] || 1) ^ 0x5eed); return Array.from({ length: 10 }, () => Math.floor(r() * 1e9)); })();
const BENCH = 'lines-10x10000+battle-send012-20k';
// Battle: 12 secret games against each bot give a steadier starting Elo.
const BATTLE_SEEDS = (() => { const r = T.rng((SEEDS[1] || 2) ^ 0xba77); return Array.from({ length: 12 }, () => Math.floor(r() * 1e9)); })();
const TAG = '[submission]';
const KEEP = 50;   // people kept on each benchmark
// Each person keeps their best agent on every benchmark: the four levels by
// lines per game and Battle by Elo. One agent can hold several of them.
const BOARDS = ['classic', 'sz', 'garbage', 'chaos', 'battle'];
const BOARD_NAMES = { classic: 'Classic', sz: 'S/Z storm', garbage: 'Rising garbage', chaos: 'Chaos', battle: 'Battle' };
const onBoard = (e, b) => (e.boards || BOARDS).includes(b);
const metric = (e, b) => b === 'battle' ? ((e.battle && e.battle.elo) || 0) : ((e.levels && e.levels[b] && e.levels[b].lines) || 0);
const K = 24;      // Elo step for games between people
const API = `https://huggingface.co/api/datasets/${REPO}`;

async function hf(path, opts = {}) {
  const r = await fetch(API + path, {
    ...opts,
    headers: { Authorization: 'Bearer ' + TOKEN, 'Content-Type': 'application/json', ...(opts.headers || {}) }
  });
  if (!r.ok) throw new Error(`${opts.method || 'GET'} ${path}: ${r.status} ${(await r.text()).slice(0, 200)}`);
  const text = await r.text();
  return text ? JSON.parse(text) : null;
}

async function openSubmissions() {
  const out = [];
  let seen = 0;
  for (let p = 0; p < 20; p++) {
    const page = await hf(`/discussions?status=open&type=discussion&p=${p}`);
    const list = page.discussions || [];
    seen += list.length;
    list.forEach(d => { if (!d.isPullRequest && String(d.title).startsWith(TAG)) out.push(d); });
    if (list.length === 0 || seen >= (page.count || 0)) break;
  }
  return out;
}

function parseAgent(raw) {
  const m = /```json\s*([\s\S]*?)```/.exec(raw || '');
  if (!m) throw new Error('No agent JSON found in the submission.');
  if (m[1].length > 64 * 1024) throw new Error('Agent JSON is too large.');
  return JSON.parse(m[1]);
}

// Only the 13 board features, bias and hidden layers. Custom rules are code,
// and this evaluator does not run other people's code.
function validate(body, fallbackName) {
  const spec = body.spec || T.defaultSpec();
  if (!Array.isArray(spec.use) || spec.use.length !== 13 || !spec.use.every(v => typeof v === 'boolean')) throw new Error('Invalid feature list.');
  if (spec.rules && spec.rules.length) throw new Error('Agents with custom rules cannot be submitted yet.');
  const clean = { use: spec.use.slice(), bias: !!spec.bias, rules: [], layers: (spec.layers || []).map(l => ({ units: l.units, act: l.act })) };
  const ev = T.makeEvaluator(clean);
  const w = body.weights;
  if (!Array.isArray(w) || w.length !== ev.size || !w.every(x => typeof x === 'number' && Number.isFinite(x) && Math.abs(x) < 1e6)) {
    throw new Error('The weights do not match the network.');
  }
  // Names keep letters, digits, spaces and - . ' only: no markdown, links, images or @mentions.
  const name = String(body.name || fallbackName || 'My agent').replace(/[^\p{L}\p{N} .'-]/gu, '').replace(/\s+/g, ' ').trim().slice(0, 24) || 'My agent';
  const arch = ev.layers.length ? ev.inputs + ' → ' + ev.layers.map(l => l.units + ' ' + l.act).join(' → ') + ' → 1' : ev.inputs + ' → 1';
  return { spec: clean, weights: w, ev, name, arch };
}

function evaluate(agent) {
  const levels = {};
  T.LEVELS.forEach(l => {
    let score = 0, lines = 0, pieces = 0;
    LEVEL_SEEDS.forEach(sd => { const r = T.play(agent.weights, sd, LEVEL_CAP, l.id, agent.ev); score += r.score; lines += r.lines; pieces += r.pieces; });
    const n = LEVEL_SEEDS.length;
    levels[l.id] = { lines: +(lines / n).toFixed(1), pieces: Math.round(pieces / n), score: Math.round(score / n) };
  });
  return levels;
}

const BOTS = require('./pool.json').agents.map(b => ({ name: b.name, elo: b.battle.elo, weights: b.weights, ev: T.makeEvaluator(b.spec || T.defaultSpec()) }));

// Battle Elo. A new agent first gets a performance rating from battles
// against the 10 bots (whose Elo is fixed, 12 games each), then plays one battle against
// each person on the Battle board. Their Elo changes are returned and only
// applied if the new agent takes a place on the Battle board.
function eloFor(agent, board, user) {
  const updates = [];
  let pts = 0, games = 0, oppElo = 0;
  BOTS.forEach(b => BATTLE_SEEDS.forEach(sd => { pts += T.battle(agent, b, sd, 'classic', CAP); games++; oppElo += b.elo; }));
  const p = Math.min(0.97, Math.max(0.03, pts / games));
  let elo = oppElo / games + 400 * Math.log10(p / (1 - p));
  board.entries.filter(e => e.user !== user && onBoard(e, 'battle')).forEach(e => {
    const other = { weights: e.weights, ev: T.makeEvaluator(e.spec) };
    const r = T.battle(agent, other, SEEDS[2], 'classic', CAP);
    const oe = (e.battle && e.battle.elo) || 1500;
    const exp = 1 / (1 + Math.pow(10, (oe - elo) / 400));
    elo += K * (r - exp);
    updates.push([e, Math.round(oe - K * (r - exp))]);
  });
  return { elo: Math.round(elo), updates };
}

// Places one evaluated submission on the board and returns the reply text.
function place(board, agent, user, d) {
  const levels = evaluate(agent);
  const { elo, updates } = eloFor(agent, board, user);
  const entry = { user, avatar: (d.author && d.author.avatarUrl) || null, name: agent.name, arch: agent.arch, levels,
    battle: { elo }, spec: agent.spec, weights: agent.weights, submitted: d.createdAt, discussion: d.num, boards: [] };
  // Take every benchmark where it beats this person's current best.
  const mine = board.entries.filter(e => e.user === user), stays = [];
  BOARDS.forEach(b => {
    const prev = mine.find(e => onBoard(e, b));
    if (!prev || metric(entry, b) > metric(prev, b)) { entry.boards.push(b); if (prev) prev.boards = prev.boards.filter(x => x !== b); }
    else stays.push(`${BOARD_NAMES[b]} (${prev.name}, ${b === 'battle' ? 'Elo ' : ''}${metric(prev, b)}${b === 'battle' ? '' : ' lines'})`);
  });
  if (entry.boards.includes('battle')) updates.forEach(([e, v]) => { e.battle = { elo: v }; });
  if (entry.boards.length) board.entries.push(entry);
  // Each benchmark keeps its top KEEP; an agent off every benchmark is dropped.
  BOARDS.forEach(b => {
    board.entries.filter(e => onBoard(e, b)).sort((x, y) => metric(y, b) - metric(x, b)).slice(KEEP)
      .forEach(e => { e.boards = e.boards.filter(x => x !== b); });
  });
  board.entries = board.entries.filter(e => e.boards.length);
  board.entries.sort((a, b) => metric(b, 'battle') - metric(a, 'battle'));
  const won = entry.boards.map(b => BOARD_NAMES[b]);
  const lines = T.LEVELS.map(l => `| ${l.name} | ${levels[l.id].lines} | ${levels[l.id].pieces} |`).join('\n');
  const why = (won.length ? `It is now your best on: ${won.join(', ')}.` : 'It did not beat your agents already on the leaderboard, or did not make a top ' + KEEP + '.')
    + (stays.length ? ` Your earlier agents keep: ${stays.join('; ')}.` : '');
  return (`**${agent.name}** by @${user} · ${agent.arch}\n\n| Level | Lines per game | Pieces per game |\n|---|---|---|\n${lines}\n\nAverages over 10 secret games of up to ${LEVEL_CAP.toLocaleString('en')} pieces per level. Battle Elo: **${elo}**. ` + why);
}

async function loadBoard() {
  const r = await fetch(`https://huggingface.co/datasets/${REPO}/resolve/main/leaderboard.json`, { headers: { Authorization: 'Bearer ' + TOKEN } });
  // Only a missing file means an empty board. Any other failure stops the run,
  // so a temporary error can never overwrite the leaderboard with nothing.
  if (!r.ok && r.status !== 404) throw new Error('could not read leaderboard.json: ' + r.status);
  const b = r.ok ? await r.json() : { entries: [] };
  if (!Array.isArray(b.entries)) b.entries = [];
  b.entries.forEach(e => { if (!Array.isArray(e.boards)) e.boards = BOARDS.slice(); });
  return b;
}

function card(board) {
  const who = e => `[@${e.user}](https://huggingface.co/${e.user}) | ${e.name.replace(/\|/g, '/')} | ${e.arch}`;
  const ranked = b => board.entries.filter(e => onBoard(e, b)).sort((x, y) => metric(y, b) - metric(x, b));
  const rows = ranked('battle').map((e, i) => `| ${i + 1} | ${who(e)} | ${metric(e, 'battle')} |`).join('\n');
  const levelTables = T.LEVELS.map(l => {
    const r = ranked(l.id).slice(0, 10).map((e, i) => `| ${i + 1} | ${who(e)} | ${metric(e, l.id)} |`).join('\n');
    return `### ${l.name}\n\n| # | User | Agent | Network | Lines per game |\n|---|---|---|---|---|\n${r || '| | No entries yet | | | |'}`;
  }).join('\n\n');
  const bots = BOTS.slice().sort((a, b) => b.elo - a.elo).map(b => `${b.name} (${b.elo})`).join(', ');
  return `---
license: mit
pretty_name: Tetris GA leaderboard
tags:
- tetris
- genetic-algorithm
---

# Tetris GA leaderboard

People's agents in [Tetris GA](https://huggingface.co/spaces/GonzaloFuentes1/tetris-ga). Each person keeps their best agent on every benchmark, and each benchmark keeps the top ${KEEP}. Updated ${board.updated.slice(0, 16).replace('T', ' ')} UTC.

## Battle

| # | User | Agent | Network | Elo |
|---|---|---|---|---|
${rows || '| | No submissions yet | | | |'}

Bots on the same Elo scale: ${bots}. Elo comes from head-to-head games against the bots and the people above.

## Levels (top 10)

Lines cleared per game, averaged over 10 secret games of up to 10,000 pieces on that level. Surviving longer and clearing more both raise it. The full lists are in the Space.

${levelTables}

## How to submit

Train an agent in the Space, sign in with Hugging Face and press *Submit to leaderboard*. That opens a discussion here titled \`[submission] <name>\`. Every 15 minutes [the arena](https://github.com/GonzaloFuentes1/tetris-ga-arena) plays new agents on secret games, so scores cannot be faked, replies with the result and closes the discussion. \`leaderboard.json\` holds every entry with its weights, so you can play against anyone in the Space.
`;
}

async function saveBoard(board) {
  board.updated = new Date().toISOString();
  const body = [
    { key: 'header', value: { summary: 'Update leaderboard', description: '' } },
    { key: 'file', value: { path: 'leaderboard.json', encoding: 'base64', content: Buffer.from(JSON.stringify(board)).toString('base64') } },
    { key: 'file', value: { path: 'README.md', encoding: 'base64', content: Buffer.from(card(board)).toString('base64') } }
  ].map(x => JSON.stringify(x)).join('\n');
  const r = await fetch(`${API}/commit/main`, {
    method: 'POST', headers: { Authorization: 'Bearer ' + TOKEN, 'Content-Type': 'application/x-ndjson' }, body
  });
  if (!r.ok) throw new Error('commit failed: ' + r.status + ' ' + (await r.text()).slice(0, 200));
}

async function reply(num, text) {
  await hf(`/discussions/${num}/comment`, { method: 'POST', body: JSON.stringify({ comment: text }) });
  await hf(`/discussions/${num}/status`, { method: 'POST', body: JSON.stringify({ status: 'closed' }) });
}

(async () => {
  if (!TOKEN) throw new Error('HF_TOKEN is not set');
  if (SEEDS.length < 3) throw new Error('EVAL_SEEDS needs at least 3 seeds');
  const subs = await openSubmissions();
  console.log(`open submissions: ${subs.length}`);
  const board = await loadBoard();
  // The job has 15 minutes; no new work starts after 10, and what is done is saved.
  const start = Date.now(), BUDGET = 10 * 60 * 1000, timeLeft = () => Date.now() - start < BUDGET;
  // A new benchmark re-measures everyone already on the board (levels, and battle
  // Elo against the bots; games between people start over), as many per run as fit.
  let changed = false;
  if (board.benchmark !== BENCH) {
    for (const e of board.entries) {
      if (e.bench === BENCH) continue;
      if (!timeLeft()) break;
      try { const a = { weights: e.weights, ev: T.makeEvaluator(e.spec) }; e.levels = evaluate(a); e.battle = { elo: eloFor(a, { entries: [] }, e.user).elo }; delete e.overall; } catch (err) { console.error('re-measure', e.user, err.message); }
      e.bench = BENCH; changed = true;
    }
    const left = board.entries.filter(e => e.bench !== BENCH).length;
    if (!left) board.benchmark = BENCH;
    console.log(`re-measured on ${BENCH}; ${left} left for the next run`);
    if (left) { await saveBoard(board); return; }
  }
  if (!subs.length) { if (changed) await saveBoard(board); return; }

  // One evaluation per user per run: the newest. Older ones close only once the
  // newest has been evaluated, so an invalid new one never costs a valid old one.
  const byUser = new Map(), older = new Map();
  subs.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt)).forEach(d => {
    const user = d.author && d.author.name;
    if (!user) return;
    if (!byUser.has(user)) { byUser.set(user, d); older.set(user, []); } else older.get(user).push(d);
  });

  const results = [];
  for (const [user, d] of byUser) {
    if (!timeLeft()) break; // the rest wait for the next run
    let full;
    try { full = await hf(`/discussions/${d.num}`); }
    catch (e) { console.error('API error, left open:', d.num, e.message); continue; }
    try {
      const first = (full.events || []).find(e => e.type === 'comment');
      const agent = validate(parseAgent(first && first.data && first.data.latest && first.data.latest.raw), String(d.title).slice(TAG.length).trim());
      results.push({ num: d.num, text: place(board, agent, user, d) });
      older.get(user).forEach(o => results.push({ num: o.num, text: `A newer submission (#${d.num}) replaces this one.` }));
    } catch (e) {
      results.push({ num: d.num, text: `Could not evaluate this submission: ${e.message}` });
    }
  }
  await saveBoard(board);
  for (const r of results) await reply(r.num, r.text).catch(e => console.error(e.message));
  console.log(`evaluated ${results.length}`);
})().catch(e => { console.error(e); process.exit(1); });
