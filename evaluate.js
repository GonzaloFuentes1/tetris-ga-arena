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
const CAP = 500;
const TAG = '[submission]';
const KEEP = 50;   // people kept on the leaderboard, by battle Elo
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
  for (let p = 0; p < 20; p++) {
    const page = await hf(`/discussions?status=open&type=discussion&p=${p}`);
    const list = page.discussions || [];
    list.forEach(d => { if (!d.isPullRequest && String(d.title).startsWith(TAG)) out.push(d); });
    if (list.length === 0 || out.length + list.length >= (page.count || 0)) break;
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
  const name = String(body.name || fallbackName || 'My agent').replace(/[\u0000-\u001f<>]/g, '').trim().slice(0, 24) || 'My agent';
  const arch = ev.layers.length ? ev.inputs + ' → ' + ev.layers.map(l => l.units + ' ' + l.act).join(' → ') + ' → 1' : ev.inputs + ' → 1';
  return { spec: clean, weights: w, ev, name, arch };
}

function evaluate(agent) {
  const levels = {};
  T.LEVELS.forEach(l => {
    let score = 0, lines = 0;
    SEEDS.forEach(sd => { const r = T.play(agent.weights, sd, CAP, l.id, agent.ev); score += r.score; lines += r.lines; });
    levels[l.id] = { score: Math.round(score / SEEDS.length), lines: +(lines / SEEDS.length).toFixed(1) };
  });
  return levels;
}

// Best average score per level among the 100 bots on the same secret seeds,
// so ratings sit on the same scale as the bot ladder.
const BOTS = require('./pool.json').agents.map(b => ({ name: b.name, elo: b.battle.elo, weights: b.weights, ev: T.makeEvaluator(b.spec || T.defaultSpec()) }));
function reference() {
  const best = {};
  T.LEVELS.forEach(l => { best[l.id] = 1; });
  BOTS.forEach(a => T.LEVELS.forEach(l => {
    let s = 0; SEEDS.forEach(sd => { s += T.play(a.weights, sd, CAP, l.id, a.ev).score; });
    best[l.id] = Math.max(best[l.id], s / SEEDS.length);
  }));
  return best;
}

// Battle Elo. A new agent first gets a performance rating from battles
// against the 10 bots (whose Elo is fixed), then plays one battle against
// each person on the board, and both sides' Elo move with every result.
function eloFor(agent, board, user) {
  let pts = 0, games = 0, oppElo = 0;
  BOTS.forEach(b => SEEDS.slice(0, 2).forEach(sd => { pts += T.battle(agent, b, sd, 'classic', CAP); games++; oppElo += b.elo; }));
  const p = Math.min(0.97, Math.max(0.03, pts / games));
  let elo = oppElo / games + 400 * Math.log10(p / (1 - p));
  board.entries.filter(e => e.user !== user).forEach(e => {
    const other = { weights: e.weights, ev: T.makeEvaluator(e.spec) };
    const r = T.battle(agent, other, SEEDS[2], 'classic', CAP);
    const oe = (e.battle && e.battle.elo) || 1500;
    const exp = 1 / (1 + Math.pow(10, (oe - elo) / 400));
    elo += K * (r - exp);
    e.battle = { elo: Math.round(oe - K * (r - exp)) };
  });
  return Math.round(elo);
}

async function loadBoard() {
  const r = await fetch(`https://huggingface.co/datasets/${REPO}/resolve/main/leaderboard.json`, { headers: { Authorization: 'Bearer ' + TOKEN } });
  const b = r.ok ? await r.json() : { entries: [] };
  if (!Array.isArray(b.entries)) b.entries = [];
  return b;
}

function card(board) {
  const rows = board.entries.map((e, i) => `| ${i + 1} | [@${e.user}](https://huggingface.co/${e.user}) | ${e.name.replace(/\|/g, '/')} | ${e.arch} | ${(e.battle && e.battle.elo) || ''} | ${e.overall} |`).join('\n');
  const bots = BOTS.slice().sort((a, b) => b.elo - a.elo).map(b => `${b.name} (${b.elo})`).join(', ');
  return `---
license: mit
pretty_name: Tetris GA leaderboard
tags:
- tetris
- genetic-algorithm
---

# Tetris GA leaderboard

People's agents in [Tetris GA](https://huggingface.co/spaces/GonzaloFuentes1/tetris-ga), ranked by battle Elo. One entry per account, top ${KEEP} only. Updated ${board.updated.slice(0, 16).replace('T', ' ')} UTC.

| # | User | Agent | Network | Battle Elo | Overall rating |
|---|---|---|---|---|---|
${rows || '| | No submissions yet | | | | |'}

Bots on the same Elo scale: ${bots}.

**Overall rating** averages the four levels (Classic, S/Z storm, Rising garbage, Chaos); 100 on a level is the best bot's average score there.

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
  if (!subs.length) return;

  // One evaluation per user per run: the newest. Older ones are closed.
  const byUser = new Map();
  subs.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt)).forEach(d => {
    const user = d.author && d.author.name;
    if (!user) return;
    if (!byUser.has(user)) byUser.set(user, d);
    else reply(d.num, `A newer submission from @${user} (#${byUser.get(user).num}) replaces this one.`).catch(e => console.error(e.message));
  });

  const best = reference();
  console.log('reference', JSON.stringify(best));
  const board = await loadBoard();
  const results = [];
  for (const [user, d] of byUser) {
    try {
      const full = await hf(`/discussions/${d.num}`);
      const first = (full.events || []).find(e => e.type === 'comment');
      const agent = validate(parseAgent(first && first.data && first.data.latest && first.data.latest.raw), String(d.title).slice(TAG.length).trim());
      const levels = evaluate(agent);
      let sum = 0;
      T.LEVELS.forEach(l => { levels[l.id].rating = +(levels[l.id].score / best[l.id] * 100).toFixed(1); sum += levels[l.id].rating; });
      const overall = +(sum / T.LEVELS.length).toFixed(1);
      const elo = eloFor(agent, board, user);
      const entry = { user, avatar: (d.author && d.author.avatarUrl) || null, name: agent.name, arch: agent.arch, levels, overall,
        battle: { elo }, spec: agent.spec, weights: agent.weights, submitted: d.createdAt, discussion: d.num };
      const prev = board.entries.find(e => e.user === user);
      let kept = !prev || elo > ((prev.battle && prev.battle.elo) || 0);
      if (kept) { if (prev) board.entries[board.entries.indexOf(prev)] = entry; else board.entries.push(entry); }
      board.entries.sort((a, b) => ((b.battle && b.battle.elo) || 0) - ((a.battle && a.battle.elo) || 0));
      if (board.entries.length > KEEP) {
        const out = board.entries.splice(KEEP);
        if (out.includes(entry)) kept = false;
      }
      const lines = T.LEVELS.map(l => `| ${l.name} | ${levels[l.id].score.toLocaleString('en')} | ${levels[l.id].lines} | ${levels[l.id].rating} |`).join('\n');
      const why = kept ? 'It is now your entry on the leaderboard.'
        : prev && prev.battle && prev.battle.elo >= elo ? `Your entry on the leaderboard has a higher Elo (${prev.battle.elo}), so it stays.`
        : `The leaderboard keeps the top ${KEEP} by battle Elo, and this one did not make it yet.`;
      results.push({ num: d.num, text: `**${agent.name}** by @${user} · ${agent.arch}\n\n| Level | Avg score | Avg lines | Rating |\n|---|---|---|---|\n${lines}\n\nBattle Elo: **${elo}**. Overall rating: **${overall}**. ` + why });
    } catch (e) {
      results.push({ num: d.num, text: `Could not evaluate this submission: ${e.message}` });
    }
  }
  await saveBoard(board);
  for (const r of results) await reply(r.num, r.text).catch(e => console.error(e.message));
  console.log(`evaluated ${results.length}`);
})().catch(e => { console.error(e); process.exit(1); });
