// Tetris, the 13 board features and the genetic algorithm of TetrisNNA-GA,
// ported from Python (TetrisAI.py, features.py, NN_GA.py) to plain JavaScript.
// Works in the browser, in a Web Worker and in Node.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.TetrisGA = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var W = 10, H = 20;
  var LINE_SCORES = [0, 40, 100, 300, 1200];

  // Same shapes as the original, as (row, col) cells.
  var SHAPES = [
    [[0, 0], [0, 1], [1, 0], [1, 1]], // O
    [[0, 0], [0, 1], [1, 1], [2, 1]], // L
    [[0, 1], [1, 1], [2, 1], [2, 0]], // J
    [[0, 1], [1, 0], [1, 1], [2, 0]], // Z
    [[0, 1], [1, 0], [1, 1], [2, 1]], // T
    [[0, 0], [1, 0], [1, 1], [2, 1]], // S
    [[0, 1], [1, 1], [2, 1], [3, 1]]  // I
  ];
  var NAMES = ['O', 'L', 'J', 'Z', 'T', 'S', 'I'];
  var GARBAGE = 8; // colour id of garbage cells

  // Difficulty levels. pieceWeights follows NAMES; garbageEvery pushes a
  // garbage row (with one gap) from the bottom every N pieces.
  var LEVELS = [
    { id: 'classic', name: 'Classic', pieceWeights: null, garbageEvery: 0 },
    { id: 'sz', name: 'S/Z storm', pieceWeights: [1, 1, 1, 4, 1, 4, 1], garbageEvery: 0 },
    { id: 'garbage', name: 'Rising garbage', pieceWeights: null, garbageEvery: 6 },
    { id: 'chaos', name: 'Chaos', pieceWeights: [1, 1, 1, 4, 1, 4, 1], garbageEvery: 6 }
  ];
  function level(id) {
    for (var i = 0; i < LEVELS.length; i++) if (LEVELS[i].id === id) return LEVELS[i];
    return LEVELS[0];
  }

  var FEATURES = [
    'agg_height', 'holes', 'bumpiness', 'pits', 'max_well', 'cols_with_holes',
    'row_transitions', 'col_transitions', 'height_variance', 'highest_peak',
    'min_height', 'hole_density', 'cleared_lines'
  ];

  // Rotation used by the original: (r, c) -> (c, size - r), then normalised.
  function rotate(cells) {
    var rs = cells.map(function (p) { return p[0]; });
    var cs = cells.map(function (p) { return p[1]; });
    var size = Math.max(Math.max.apply(null, rs) - Math.min.apply(null, rs),
                        Math.max.apply(null, cs) - Math.min.apply(null, cs));
    return cells.map(function (p) { return [p[1], size - p[0]]; });
  }
  function normalise(cells) {
    var mr = Math.min.apply(null, cells.map(function (p) { return p[0]; }));
    var mc = Math.min.apply(null, cells.map(function (p) { return p[1]; }));
    return cells.map(function (p) { return [p[0] - mr, p[1] - mc]; })
      .sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
  }
  // Distinct orientations of each piece.
  var ROTATIONS = SHAPES.map(function (shape) {
    var out = [], seen = {}, cur = shape;
    for (var i = 0; i < 4; i++) {
      var n = normalise(cur), key = JSON.stringify(n);
      if (!seen[key]) { seen[key] = true; out.push(n); }
      cur = rotate(cur);
    }
    return out;
  });

  // Small seeded RNG so every agent of a generation sees the same pieces.
  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // The 13 features of TetrisAI.calculate_features, on a 0/1 board
  // (row 0 on top) that already holds the placed piece, before clearing lines.
  function features(b) {
    var peaks = new Array(W), c, r;
    for (c = 0; c < W; c++) {
      peaks[c] = 0;
      for (r = 0; r < H; r++) if (b[r * W + c]) { peaks[c] = H - r; break; }
    }
    var highest = 0, agg = 0, minH = Infinity;
    for (c = 0; c < W; c++) {
      if (peaks[c] > highest) highest = peaks[c];
      agg += peaks[c];
      if (peaks[c] < minH) minH = peaks[c];
    }
    var nHoles = 0, colsWithHoles = 0, pits = 0, blocks = 0;
    for (c = 0; c < W; c++) {
      var h = 0, filled = 0;
      for (r = 0; r < H; r++) if (b[r * W + c]) filled++;
      if (filled === 0) pits++;
      blocks += filled;
      if (peaks[c] > 0) {
        for (r = H - peaks[c]; r < H; r++) if (!b[r * W + c]) h++;
      }
      nHoles += h;
      if (h > 0) colsWithHoles++;
    }
    var rowT = 0;
    for (r = H - highest; r < H; r++) {
      for (c = 1; c < W; c++) if (b[r * W + c] !== b[r * W + c - 1]) rowT++;
    }
    var colT = 0;
    for (c = 0; c < W; c++) {
      if (peaks[c] <= 1) continue;
      for (r = H - peaks[c]; r < H - 1; r++) if (b[r * W + c] !== b[(r + 1) * W + c]) colT++;
    }
    var bump = 0;
    for (c = 0; c < W - 1; c++) bump += Math.abs(peaks[c] - peaks[c + 1]);
    var maxWell = 0;
    for (c = 0; c < W; c++) {
      var left = c > 0 ? peaks[c - 1] - peaks[c] : 0;
      var right = c < W - 1 ? peaks[c + 1] - peaks[c] : 0;
      var w = Math.max(left, right, 0);
      if (w > maxWell) maxWell = w;
    }
    var mean = agg / W, variance = 0;
    for (c = 0; c < W; c++) variance += (peaks[c] - mean) * (peaks[c] - mean);
    variance /= W;
    var cleared = 0;
    for (r = 0; r < H; r++) {
      var full = true;
      for (c = 0; c < W; c++) if (!b[r * W + c]) { full = false; break; }
      if (full) cleared++;
    }
    var density = blocks > 0 ? nHoles / blocks : 0;
    return [agg, nHoles, bump, pits, maxWell, colsWithHoles, rowT, colT,
            variance, highest, minH, density, cleared];
  }

  // Every landing spot of the piece: one per orientation and column.
  function placements(board, piece) {
    var out = [];
    ROTATIONS[piece].forEach(function (cells, rot) {
      var width = 1 + Math.max.apply(null, cells.map(function (p) { return p[1]; }));
      var height = 1 + Math.max.apply(null, cells.map(function (p) { return p[0]; }));
      for (var col = 0; col + width <= W; col++) {
        var row = -height;
        while (fits(board, cells, row + 1, col)) row++;
        var ok = cells.every(function (p) { return p[0] + row >= 0; });
        if (ok) out.push({ rot: rot, col: col, row: row, cells: cells });
      }
    });
    return out;
  }
  function fits(board, cells, row, col) {
    for (var i = 0; i < cells.length; i++) {
      var r = cells[i][0] + row, c = cells[i][1] + col;
      if (r >= H) return false;
      if (r >= 0 && board[r * W + c]) return false;
    }
    return true;
  }

  function score(weights, f) {
    var s = 0;
    for (var i = 0; i < f.length; i++) s += weights[i] * f[i];
    return s;
  }

  // One game, played by a linear evaluator over the 13 features.
  function Game(seed, levelId) {
    this.lvl = level(levelId);
    this.board = new Uint8Array(W * H);
    this.rand = rng(seed);
    this.garbageRand = rng(seed ^ 0x9E3779B9);
    this.score = 0; this.lines = 0; this.level = 0; this.pieces = 0;
    this.over = false;
    this.next();
  }
  Game.prototype.next = function () {
    var pw = this.lvl.pieceWeights;
    if (!pw) this.piece = Math.floor(this.rand() * SHAPES.length);
    else {
      var total = 0, i, x;
      for (i = 0; i < pw.length; i++) total += pw[i];
      x = this.rand() * total;
      for (i = 0; i < pw.length; i++) { x -= pw[i]; if (x < 0) break; }
      this.piece = Math.min(i, pw.length - 1);
    }
    this.color = 1 + this.piece;
  };
  // Pushes a garbage row with one random gap from the bottom.
  Game.prototype.garbage = function () {
    var b = this.board, c;
    for (c = 0; c < W; c++) if (b[c]) { this.over = true; return; }
    b.copyWithin(0, W);
    var gap = Math.floor(this.garbageRand() * W);
    for (c = 0; c < W; c++) b[(H - 1) * W + c] = c === gap ? 0 : GARBAGE;
  };
  // Battle mode: the opponent's clears arrive here as garbage rows.
  Game.prototype.receive = function (n) {
    for (var i = 0; i < n && !this.over; i++) this.garbage();
  };
  // Picks the best spot for the current piece and places it. brain is either
  // a feature function (scored linearly) or an evaluator from makeEvaluator;
  // by default the 13 original features with a linear score.
  Game.prototype.step = function (weights, brain) {
    var ev = brain ? (typeof brain === 'function' ? brain : brain.fn) : features;
    var val = brain && brain.value ? brain.value : score;
    if (this.over) return null;
    var opts = placements(this.board, this.piece);
    if (!opts.length) { this.over = true; return null; }
    var best = null, bestV = -Infinity, bestF = null, bin = new Uint8Array(W * H), i, k;
    for (i = 0; i < opts.length; i++) {
      for (k = 0; k < W * H; k++) bin[k] = this.board[k] ? 1 : 0;
      var o = opts[i];
      o.cells.forEach(function (p) { bin[(p[0] + o.row) * W + p[1] + o.col] = 1; });
      var f = ev(bin), v = val(weights, f);
      if (v > bestV || best === null) { bestV = v; best = o; bestF = f; }
    }
    var color = this.color, board = this.board;
    best.cells.forEach(function (p) { board[(p[0] + best.row) * W + p[1] + best.col] = color; });
    var kept = [], cleared = 0, r, c;
    for (r = 0; r < H; r++) {
      var full = true;
      for (c = 0; c < W; c++) if (!board[r * W + c]) { full = false; break; }
      if (full) cleared++; else kept.push(board.slice(r * W, r * W + W));
    }
    if (cleared) {
      board.fill(0);
      for (r = 0; r < kept.length; r++) board.set(kept[r], (cleared + r) * W);
      this.lines += cleared;
      this.score += LINE_SCORES[cleared] * (this.level + 1);
      this.level = Math.floor(this.lines / 10);
    }
    this.pieces++;
    var ge = this.lvl.garbageEvery;
    if (ge && this.pieces % ge === 0) this.garbage();
    this.next();
    // feats and value describe the chosen move, for drawing the network.
    return { placed: best, cleared: cleared, feats: bestF, value: bestV };
  };

  function play(weights, seed, maxPieces, levelId, brain) {
    var g = new Game(seed, levelId);
    while (!g.over && g.pieces < maxPieces) g.step(weights, brain);
    return { score: g.score, lines: g.lines, pieces: g.pieces, over: g.over };
  }

  // Battle: both agents get the same pieces; clearing 3 lines at once sends 1
  // garbage row to the other board and a tetris (4 lines) sends 2. The last
  // one standing wins; if both reach maxPieces, the higher score wins.
  // a and b are { weights, ev }. Returns 1 if a wins, 0 if b wins, 0.5 on a draw.
  var SEND = [0, 0, 0, 1, 2];
  function battle(a, b, seed, levelId, maxPieces) {
    var ga = new Game(seed, levelId), gb = new Game(seed, levelId);
    while (true) {
      var ra = ga.over || ga.pieces >= maxPieces ? null : ga.step(a.weights, a.ev);
      var rb = gb.over || gb.pieces >= maxPieces ? null : gb.step(b.weights, b.ev);
      if (ra && SEND[ra.cleared]) gb.receive(SEND[ra.cleared]);
      if (rb && SEND[rb.cleared]) ga.receive(SEND[rb.cleared]);
      if (ga.over !== gb.over) return ga.over ? 0 : 1;
      var done = (ga.over || ga.pieces >= maxPieces) && (gb.over || gb.pieces >= maxPieces);
      if (done) return ga.score > gb.score ? 1 : ga.score < gb.score ? 0 : 0.5;
    }
  }

  // Genetic algorithm of NN_GA.py: fitness-proportional parents, uniform
  // crossover and Gaussian mutation. One change: the elite is copied without
  // mutation (the original mutated it too).
  function randomWeights(rand, lo, hi, n) {
    var w = [], len = n || FEATURES.length;
    for (var i = 0; i < len; i++) w.push(lo + (hi - lo) * rand());
    return w;
  }
  function gauss(rand) {
    var u = 1 - rand(), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function pickTwo(rand, probs) {
    function pick(exclude) {
      var total = 0, i;
      for (i = 0; i < probs.length; i++) if (i !== exclude) total += probs[i];
      var x = rand() * total;
      for (i = 0; i < probs.length; i++) {
        if (i === exclude) continue;
        x -= probs[i];
        if (x <= 0) return i;
      }
      return exclude === probs.length - 1 ? probs.length - 2 : probs.length - 1;
    }
    var a = pick(-1);
    return [a, pick(a)];
  }
  // Options (defaults reproduce the original):
  //   selection: 'roulette' (proportional to fitness), 'tournament' (best of 3) or 'rank'
  //   crossover: 'uniform', 'one_point', 'two_point', 'blend' or 'none'
  //   crossoverRate: chance a child mixes two parents; otherwise it copies one
  var SELECTIONS = ['roulette', 'tournament', 'rank'];
  var CROSSOVERS = ['uniform', 'one_point', 'two_point', 'blend', 'none'];
  function tournament(rand, fitness, exclude) {
    var best = -1;
    for (var k = 0; k < 3; k++) {
      var c = Math.floor(rand() * fitness.length);
      if (c === exclude) c = (c + 1) % fitness.length;
      if (best < 0 || fitness[c] > fitness[best]) best = c;
    }
    return best;
  }
  function cross(a, b, kind, rand) {
    var len = a.length, child = new Array(len), j;
    if (kind === 'one_point' || kind === 'two_point') {
      var p1 = Math.floor(rand() * len), p2 = kind === 'two_point' ? Math.floor(rand() * len) : len;
      if (p2 < p1) { var t = p1; p1 = p2; p2 = t; }
      for (j = 0; j < len; j++) child[j] = j >= p1 && j < p2 ? b[j] : a[j];
    } else if (kind === 'blend') {
      for (j = 0; j < len; j++) { var al = rand(); child[j] = al * a[j] + (1 - al) * b[j]; }
    } else if (kind === 'none') {
      for (j = 0; j < len; j++) child[j] = a[j];
    } else {
      for (j = 0; j < len; j++) child[j] = rand() < 0.5 ? a[j] : b[j];
    }
    return child;
  }
  function nextGeneration(pop, fitness, opts, rand) {
    var n = pop.length, sel = opts.selection || 'roulette', kind = opts.crossover || 'uniform';
    var rate = opts.crossoverRate === undefined ? 1 : opts.crossoverRate;
    var order = pop.map(function (_, i) { return i; })
      .sort(function (a, b) { return fitness[b] - fitness[a]; });
    var probs;
    if (sel === 'rank') {
      // Linear ranking: the best gets weight n, the worst 1.
      var rankW = new Array(n);
      order.forEach(function (idx, r) { rankW[idx] = n - r; });
      var rs = n * (n + 1) / 2;
      probs = rankW.map(function (x) { return x / rs; });
    } else {
      var total = fitness.reduce(function (s, f) { return s + Math.max(0, f); }, 0);
      probs = fitness.map(function (f) { return total > 0 ? Math.max(0, f) / total + 1e-14 : 1 / n; });
    }
    var nElite = Math.min(n, Math.round(n * (opts.elitism || 0)));
    var out = [], i, j;
    for (i = 0; i < nElite && i < n; i++) out.push(pop[order[i]].slice());
    while (out.length < n) {
      var ab;
      if (sel === 'tournament') { var x = tournament(rand, fitness, -1); ab = [x, tournament(rand, fitness, x)]; }
      else ab = pickTwo(rand, probs);
      var mix = rate >= 1 || rand() < rate;
      var child = mix ? cross(pop[ab[0]], pop[ab[1]], kind, rand) : pop[ab[0]].slice(), len = child.length;
      for (j = 0; j < len; j++) {
        if (rand() < opts.mutationProb) child[j] += gauss(rand) * opts.mutationPower;
      }
      out.push(child);
    }
    return out;
  }

  // ---------- Custom agents ----------
  // spec = { use: [13 booleans], bias: bool, rules: [{ name, code }] }.
  // A rule is the body of a function (b, W, H, heights, cell) that returns a
  // number: b is the 0/1 board (row 0 on top), heights the column heights and
  // cell(r, c) reads a cell (0 outside the board).
  // At most 20 features in total: chosen base features + rules + bias.
  var MAX_FEATURES = 20, MAX_RULE_LINES = 12, MAX_RULE_CHARS = 600;
  function compileRule(code) {
    var lines = code.split('\n').length;
    if (lines > MAX_RULE_LINES) throw new Error('Rules can have at most ' + MAX_RULE_LINES + ' lines (this one has ' + lines + ').');
    if (code.length > MAX_RULE_CHARS) throw new Error('Rules can have at most ' + MAX_RULE_CHARS + ' characters.');
    return new Function('b', 'W', 'H', 'heights', 'cell', '"use strict";\n' + code);
  }
  function defaultSpec() {
    return { use: FEATURES.map(function () { return true; }), bias: false, rules: [], layers: [] };
  }
  function isDefault(spec) {
    return !spec || (!spec.bias && !(spec.rules || []).length && !(spec.layers || []).length && spec.use.every(Boolean));
  }

  // ---------- Network ----------
  // Without hidden layers the agent is the original linear model. With them
  // it is a small MLP: up to 3 hidden layers of 1..16 units, one output. The
  // GA evolves every weight and bias. Inputs are divided by a typical scale
  // so tanh and sigmoid do not saturate.
  var MAX_LAYERS = 3, MAX_UNITS = 16;
  // exp and tanh built only from + - * /, which give the same bits in every
  // JavaScript engine. Math.exp and Math.tanh may differ in the last digit
  // between browsers and Node, and over a long game that changes moves.
  var LN2_HI = 0.6931471803691238, LN2_LO = 1.9082149292705877e-10;
  function dexp(x) {
    if (x !== x) return x;
    if (x > 700) x = 700; else if (x < -700) return 0;
    var n = Math.round(x / 0.6931471805599453), r = (x - n * LN2_HI) - n * LN2_LO;
    // Taylor series on |r| <= 0.35, 14 terms.
    var p = 1 / 87178291200;
    for (var k = 13; k >= 1; k--) p = p * r + 1 / FACT[k];
    p = p * r + 1;
    var s = 1, b = n >= 0 ? 2 : 0.5;
    for (var i = Math.abs(n); i > 0; i--) s *= b;
    return p * s;
  }
  var FACT = [1];
  for (var fi = 1; fi <= 14; fi++) FACT[fi] = FACT[fi - 1] * fi;
  function dtanh(x) {
    if (x !== x) return x;
    if (x > 20) return 1; if (x < -20) return -1;
    var t = dexp(-2 * Math.abs(x)), y = (1 - t) / (1 + t);
    return x < 0 ? -y : y;
  }
  var ACTS = {
    relu: function (x) { return x > 0 ? x : 0; },
    leaky_relu: function (x) { return x > 0 ? x : 0.05 * x; },
    tanh: dtanh,
    sigmoid: function (x) { return 1 / (1 + dexp(-x)); },
    elu: function (x) { return x > 0 ? x : dexp(x) - 1; }
  };
  var SCALES = [200, 40, 60, 10, 20, 10, 80, 80, 40, 20, 20, 1, 4];
  function checkLayers(layers) {
    if (layers.length > MAX_LAYERS) throw new Error('At most ' + MAX_LAYERS + ' hidden layers.');
    layers.forEach(function (l) {
      if (!(l.units >= 1 && l.units <= MAX_UNITS && l.units === Math.floor(l.units))) throw new Error('Each hidden layer needs 1 to ' + MAX_UNITS + ' units.');
      if (!Object.prototype.hasOwnProperty.call(ACTS, l.act)) throw new Error('Unknown activation: ' + l.act);
    });
  }
  function paramCount(inputs, layers) {
    var n = 0, prev = inputs;
    layers.forEach(function (l) { n += prev * l.units + l.units; prev = l.units; });
    return n + prev;
  }
  // Returns { out, acts } where acts[k] are the activations of layer k
  // (acts[0] are the scaled inputs).
  function forward(weights, x, layers) {
    var acts = [x], off = 0, cur = x;
    for (var k = 0; k < layers.length; k++) {
      var L = layers[k], f = ACTS[L.act], nIn = cur.length, next = new Array(L.units);
      for (var j = 0; j < L.units; j++) {
        var s = weights[off + nIn * L.units + j];
        for (var i = 0; i < nIn; i++) s += weights[off + j * nIn + i] * cur[i];
        next[j] = f(s);
      }
      off += nIn * L.units + L.units; cur = next; acts.push(cur);
    }
    var out = 0;
    for (var m = 0; m < cur.length; m++) out += weights[off + m] * cur[m];
    return { out: out, acts: acts };
  }

  function makeEvaluator(spec) {
    var layers = (spec && spec.layers) || [];
    checkLayers(layers);
    if (isDefault(spec)) {
      var dk = FEATURES.map(function (_, i) { return 'f' + i; });
      return { fn: features, value: score, size: FEATURES.length, inputs: FEATURES.length, keys: dk, layers: [],
        forward: function (w, f) { return { out: score(w, f), acts: [f] }; } };
    }
    var idx = [], keys = [], i;
    for (i = 0; i < FEATURES.length; i++) if (spec.use[i]) { idx.push(i); keys.push('f' + i); }
    var rules = (spec.rules || []).map(function (r, k) { keys.push('r' + k); return compileRule(r.code); });
    if (spec.bias) keys.push('bias');
    if (keys.length > MAX_FEATURES) throw new Error('At most ' + MAX_FEATURES + ' features in total; this agent has ' + keys.length + '.');
    if (!keys.length) throw new Error('Pick at least one feature.');
    var heights = new Array(W);
    function cellOf(b) { return function (r, c) { return r >= 0 && r < H && c >= 0 && c < W ? b[r * W + c] : 0; }; }
    function fn(b) {
      var f = features(b), out = [], k, c, r;
      for (k = 0; k < idx.length; k++) out.push(f[idx[k]]);
      if (rules.length) {
        for (c = 0; c < W; c++) {
          heights[c] = 0;
          for (r = 0; r < H; r++) if (b[r * W + c]) { heights[c] = H - r; break; }
        }
        var cell = cellOf(b);
        for (k = 0; k < rules.length; k++) {
          var v = +rules[k](b, W, H, heights.slice(), cell);
          out.push(isFinite(v) ? v : 0);
        }
      }
      if (spec.bias) out.push(1);
      return out;
    }
    if (!layers.length) {
      return { fn: fn, value: score, size: keys.length, inputs: keys.length, keys: keys, layers: [],
        forward: function (w, f) { return { out: score(w, f), acts: [f] }; } };
    }
    var scale = keys.map(function (k) { return k[0] === 'f' ? SCALES[+k.slice(1)] : 1; });
    function scaled(f) { var x = new Array(f.length); for (var q = 0; q < f.length; q++) x[q] = f[q] / scale[q]; return x; }
    return {
      fn: fn, keys: keys, inputs: keys.length, layers: layers, size: paramCount(keys.length, layers),
      value: function (w, f) { return forward(w, scaled(f), layers).out; },
      forward: function (w, f) { return forward(w, scaled(f), layers); }
    };
  }

  return {
    MAX_FEATURES: MAX_FEATURES, MAX_RULE_LINES: MAX_RULE_LINES, MAX_LAYERS: MAX_LAYERS, MAX_UNITS: MAX_UNITS,
    ACTIVATIONS: Object.keys(ACTS), paramCount: paramCount, compileRule: compileRule,
    defaultSpec: defaultSpec, makeEvaluator: makeEvaluator,
    W: W, H: H, LEVELS: LEVELS, SHAPES: SHAPES, NAMES: NAMES, ROTATIONS: ROTATIONS, FEATURES: FEATURES,
    rng: rng, dexp: dexp, dtanh: dtanh, features: features, placements: placements, Game: Game, play: play, battle: battle, SEND: SEND,
    randomWeights: randomWeights, nextGeneration: nextGeneration, SELECTIONS: SELECTIONS, CROSSOVERS: CROSSOVERS
  };
});
