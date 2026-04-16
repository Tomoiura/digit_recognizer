"""Generate self-contained HTML with embedded CNN weights and JS inference."""

import json


def build_html(weights: dict, history: dict, test_acc: float) -> str:
    weights_json = json.dumps(weights)

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>手書き数字認識 AI</title>
<style>
{_css()}
</style>
</head>
<body>

<header>
  <h1>手書き数字認識 AI</h1>
  <p class="subtitle">指またはマウスで数字を描くと、ニューラルネットワークがリアルタイムで認識します</p>
  <p class="subtitle2">同時に、ネットワーク内部の信号の流れがリアルタイムに可視化されます</p>
  <p class="model-info">学習済みモデル &#9432;
    <span class="tooltip" id="model-tooltip">MNIST（手書き数字の画像7万枚のデータセット）のうち<br>6万枚で学習し、残り1万枚でテストした正解率: {test_acc:.1%}<br>モデル構成: Conv(5x5,8ch) → Pool → Conv(3x3,16ch) → Pool → FC(64) → FC(10)<br><br>小さいながらも27,690パラメータ（GPT-4の6,500万分の1）を持つ<br>本物のニューラルネットワークです。<br>それでも98%の精度が出せるのでCNNの効率の良さがわかります。<br><button class="copy-btn" onclick="copyModelInfo(event)">コピー</button></span>
  </p>
</header>

<main>
  <!-- Upper section: dial + canvas -->
  <section id="upper">
    <div id="dial-wrap">
      <div id="dial">
        <div class="dial-cell" data-digit="1"><span>1</span></div>
        <div class="dial-cell" data-digit="2"><span>2</span></div>
        <div class="dial-cell" data-digit="3"><span>3</span></div>
        <div class="dial-cell" data-digit="4"><span>4</span></div>
        <div class="dial-cell" data-digit="5"><span>5</span></div>
        <div class="dial-cell" data-digit="6"><span>6</span></div>
        <div class="dial-cell" data-digit="7"><span>7</span></div>
        <div class="dial-cell" data-digit="8"><span>8</span></div>
        <div class="dial-cell" data-digit="9"><span>9</span></div>
        <div class="dial-cell empty"></div>
        <div class="dial-cell" data-digit="0"><span>0</span></div>
        <div class="dial-cell empty"></div>
      </div>
    </div>
    <div id="canvas-wrap">
      <canvas id="draw-canvas" width="280" height="280"></canvas>
      <div id="canvas-controls">
        <button id="btn-clear">Clear</button>
        <button id="btn-swap">&#8644; Left / Right</button>
      </div>
      <div id="cnn-input-preview">
        <canvas id="vis-input" width="28" height="28"></canvas>
        <span>CNNの入力 (28&times;28)</span>
      </div>
    </div>
  </section>

  <!-- Network diagram -->
  <section id="nn-diagram">
    <h2>Network Diagram</h2>
    <svg id="nn-svg" viewBox="0 0 840 440" preserveAspectRatio="xMidYMid meet">
      <defs>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <g id="nn-links"></g>
      <g id="nn-nodes"></g>
      <g id="nn-labels"></g>
    </svg>
  </section>

</main>

<script>
// ===================== WEIGHTS =====================
const W = {weights_json};

// ===================== CNN INFERENCE IN JS =====================
{_js_inference()}

// ===================== DRAWING + UI =====================
{_js_ui()}
</script>
</body>
</html>"""


def _css() -> str:
    return """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

:root {
  --bg: #0d1117;
  --surface: #161b22;
  --border: #30363d;
  --text: #e6edf3;
  --text-dim: #8b949e;
  --accent: #ff6b35;
  --accent-glow: rgba(255,107,53,0.4);
  --cool: #58a6ff;
  --hot: #ff4444;
  --warm: #ff8c00;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  overflow-x: hidden;
}

header {
  text-align: center;
  padding: 28px 16px 20px;
}
header h1 {
  font-size: 2rem;
  font-weight: 700;
  color: var(--accent);
}
.subtitle {
  font-size: 1.05rem;
  color: var(--text);
  margin-top: 8px;
}
.meta {
  font-size: 0.8rem;
  color: var(--text);
  margin-top: 6px;
  opacity: 0.6;
}

.subtitle2 {
  font-size: 1.05rem;
  color: var(--text);
  margin-top: 4px;
}

.model-info {
  display: inline-block;
  font-size: 0.9rem;
  color: var(--text);
  margin-top: 10px;
  cursor: help;
  border-bottom: 1px dashed var(--text);
  padding-bottom: 1px;
  position: relative;
}
.model-info .tooltip {
  display: none;
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  top: calc(100% + 8px);
  background: #2a3a4a;
  color: var(--text);
  font-size: 0.8rem;
  line-height: 1.6;
  padding: 12px 16px;
  border-radius: 8px;
  border: 1px solid var(--border);
  white-space: nowrap;
  z-index: 10;
  box-shadow: 0 4px 16px rgba(0,0,0,0.5);
}
.model-info.open .tooltip {
  display: block;
}
.copy-btn {
  margin-top: 8px;
  padding: 4px 12px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--surface);
  color: var(--text);
  font-size: 0.75rem;
  cursor: pointer;
}
.copy-btn:hover {
  background: var(--border);
}

main {
  max-width: 900px;
  margin: 0 auto;
  padding: 0 16px 40px;
}

/* ---- Upper section: dial + canvas ---- */
#upper {
  display: flex;
  gap: 16px;
  justify-content: center;
  align-items: flex-start;
  margin-bottom: 28px;
}
#upper.swapped { flex-direction: row-reverse; }

/* Dial */
#dial-wrap {
  flex-shrink: 0;
}
#dial {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
  width: 160px;
}
.dial-cell {
  aspect-ratio: 1;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.5rem;
  font-weight: 700;
  background: var(--surface);
  border: 2px solid var(--border);
  position: relative;
  overflow: hidden;
  transition: background 0.15s, border-color 0.15s, box-shadow 0.15s;
}
.dial-cell.empty {
  visibility: hidden;
}
.dial-cell span {
  position: relative;
  z-index: 1;
  text-shadow: 0 0 8px rgba(0,0,0,0.8);
}
.dial-cell::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 10px;
  opacity: 0;
  transition: opacity 0.15s;
}
.dial-cell[data-heat] {
  /* heat is set via JS inline style */
}

/* Canvas */
#canvas-wrap {
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}
#draw-canvas {
  width: 220px;
  height: 220px;
  background:
    linear-gradient(rgba(255,255,255,0.25) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.25) 1px, transparent 1px),
    linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px),
    #3d5060;
  background-size:
    calc(220px / 4) calc(220px / 4),
    calc(220px / 4) calc(220px / 4),
    calc(220px / 28) calc(220px / 28),
    calc(220px / 28) calc(220px / 28);
  border: 2px solid var(--cool);
  border-radius: 12px;
  cursor: crosshair;
  touch-action: none;
  box-shadow: 0 0 16px rgba(88,166,255,0.2);
}
#canvas-controls {
  display: flex;
  gap: 8px;
}
#canvas-controls button {
  padding: 6px 14px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text);
  font-size: 0.8rem;
  cursor: pointer;
  transition: background 0.15s;
}
#canvas-controls button:hover {
  background: var(--border);
}

/* ---- Network Diagram ---- */
#nn-diagram {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 24px;
}
#nn-diagram h2 {
  font-size: 1.1rem;
  color: var(--accent);
  margin-bottom: 10px;
  text-align: center;
}
#nn-svg {
  width: 100%;
  height: auto;
  max-height: 360px;
}

/* CNN Input Preview */
#cnn-input-preview {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 4px;
}
#cnn-input-preview canvas {
  width: 42px;
  height: 42px;
  border-radius: 4px;
  border: 1px solid var(--border);
  image-rendering: pixelated;
  background: #000;
}
#cnn-input-preview span {
  font-size: 0.95rem;
  color: var(--text);
}

/* Responsive */
@media (max-width: 500px) {
  #upper { gap: 10px; }
  #dial { width: 130px; gap: 4px; }
  .dial-cell { font-size: 1.2rem; border-radius: 10px; }
  #draw-canvas { width: 180px; height: 180px; }
  #flow-container { gap: 4px; }
}
"""


def _js_inference() -> str:
    return """
function makeNdArray(flat, shape) {
  return { data: new Float32Array(flat), shape: shape };
}

function conv2d(input, weight, bias) {
  // input: {data, shape:[N,C,H,W]}, weight: {data, shape:[Co,Ci,kH,kW]}
  const [N, Ci, H, Wi] = input.shape;
  const [Co, , kH, kW] = weight.shape;
  const Ho = H - kH + 1;
  const Wo = Wi - kW + 1;
  const out = new Float32Array(N * Co * Ho * Wo);

  for (let n = 0; n < N; n++) {
    for (let co = 0; co < Co; co++) {
      for (let oh = 0; oh < Ho; oh++) {
        for (let ow = 0; ow < Wo; ow++) {
          let sum = bias[co];
          for (let ci = 0; ci < Ci; ci++) {
            for (let kh = 0; kh < kH; kh++) {
              for (let kw = 0; kw < kW; kw++) {
                const ii = ((n * Ci + ci) * H + (oh + kh)) * Wi + (ow + kw);
                const wi = ((co * Ci + ci) * kH + kh) * kW + kw;
                sum += input.data[ii] * weight.data[wi];
              }
            }
          }
          out[((n * Co + co) * Ho + oh) * Wo + ow] = sum;
        }
      }
    }
  }
  return { data: out, shape: [N, Co, Ho, Wo] };
}

function relu(input) {
  const out = new Float32Array(input.data.length);
  for (let i = 0; i < out.length; i++) {
    out[i] = input.data[i] > 0 ? input.data[i] : 0;
  }
  return { data: out, shape: input.shape.slice() };
}

function maxpool2d(input, size) {
  const [N, C, H, Wi] = input.shape;
  const Ho = Math.floor(H / size);
  const Wo = Math.floor(Wi / size);
  const out = new Float32Array(N * C * Ho * Wo);
  for (let n = 0; n < N; n++) {
    for (let c = 0; c < C; c++) {
      for (let oh = 0; oh < Ho; oh++) {
        for (let ow = 0; ow < Wo; ow++) {
          let mx = -Infinity;
          for (let sh = 0; sh < size; sh++) {
            for (let sw = 0; sw < size; sw++) {
              const idx = ((n * C + c) * H + oh * size + sh) * Wi + ow * size + sw;
              if (input.data[idx] > mx) mx = input.data[idx];
            }
          }
          out[((n * C + c) * Ho + oh) * Wo + ow] = mx;
        }
      }
    }
  }
  return { data: out, shape: [N, C, Ho, Wo] };
}

function fc(input, weight, bias, inDim, outDim) {
  // input.data length = N*inDim, weight: flat [inDim, outDim]
  const N = input.data.length / inDim;
  const out = new Float32Array(N * outDim);
  for (let n = 0; n < N; n++) {
    for (let o = 0; o < outDim; o++) {
      let sum = bias[o];
      for (let i = 0; i < inDim; i++) {
        sum += input.data[n * inDim + i] * weight[i * outDim + o];
      }
      out[n * outDim + o] = sum;
    }
  }
  return { data: out, shape: [N, outDim] };
}

function softmax(logits) {
  const n = logits.length;
  let mx = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] > mx) mx = logits[i];
  const ex = new Float32Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) { ex[i] = Math.exp(logits[i] - mx); sum += ex[i]; }
  for (let i = 0; i < n; i++) ex[i] /= sum;
  return ex;
}

// Prepare weight tensors
function loadWeights(W) {
  const flat = (arr) => new Float32Array(arr.flat(Infinity));
  return {
    conv1_W: { data: flat(W.conv1_W), shape: [8, 1, 5, 5] },
    conv1_b: flat(W.conv1_b),
    conv2_W: { data: flat(W.conv2_W), shape: [16, 8, 3, 3] },
    conv2_b: flat(W.conv2_b),
    fc1_W: flat(W.fc1_W),
    fc1_b: flat(W.fc1_b),
    fc2_W: flat(W.fc2_W),
    fc2_b: flat(W.fc2_b),
  };
}

function infer(pixels28, weights) {
  // pixels28: Float32Array(784), 0..1
  const input = { data: pixels28, shape: [1, 1, 28, 28] };

  const c1 = conv2d(input, weights.conv1_W, weights.conv1_b);
  const r1 = relu(c1);         // [1, 8, 24, 24]
  const p1 = maxpool2d(r1, 2); // [1, 8, 12, 12]

  const c2 = conv2d(p1, weights.conv2_W, weights.conv2_b);
  const r2 = relu(c2);         // [1, 16, 10, 10]
  const p2 = maxpool2d(r2, 2); // [1, 16, 5, 5]

  const fc1out = fc(p2, weights.fc1_W, weights.fc1_b, 400, 64);
  const fc1r = relu(fc1out);

  const fc2out = fc(fc1r, weights.fc2_W, weights.fc2_b, 64, 10);
  const probs = softmax(fc2out.data);

  return {
    input: pixels28,
    conv1: r1,    // 8 @ 24x24
    pool1: p1,    // 8 @ 12x12
    conv2: r2,    // 16 @ 10x10
    pool2: p2,    // 16 @ 5x5
    fc1: fc1r,    // 64
    logits: fc2out,
    probs: probs, // 10
  };
}
"""


def _js_ui() -> str:
    return """
// ===================== INIT =====================
const weights = loadWeights(W);
const canvas = document.getElementById('draw-canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let hasContent = false;

// High-res canvas
const ratio = 2;
canvas.width = canvas.offsetWidth * ratio;
canvas.height = canvas.offsetHeight * ratio;
ctx.scale(ratio, ratio);
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = '#ffffff';
ctx.lineWidth = 14;

// ===================== DRAWING =====================
function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const t = e.touches ? e.touches[0] : e;
  return {
    x: (t.clientX - rect.left),
    y: (t.clientY - rect.top),
  };
}

canvas.addEventListener('pointerdown', (e) => {
  isDrawing = true;
  hasContent = true;
  const p = getPos(e);
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
  e.preventDefault();
});

canvas.addEventListener('pointermove', (e) => {
  if (!isDrawing) return;
  const p = getPos(e);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
  runInference();
  e.preventDefault();
});

canvas.addEventListener('pointerup', () => { isDrawing = false; runInference(); });
canvas.addEventListener('pointerleave', () => { isDrawing = false; });

// ===================== CLEAR =====================
document.getElementById('btn-clear').addEventListener('click', () => {
  ctx.clearRect(0, 0, canvas.width / ratio, canvas.height / ratio);
  hasContent = false;
  updateDial(new Float32Array(10).fill(0.1));
  clearInputPreview();
  clearNetworkDiagram();
});

// ===================== SWAP =====================
document.getElementById('btn-swap').addEventListener('click', () => {
  document.getElementById('upper').classList.toggle('swapped');
});

// ===================== INFERENCE =====================
function getPixels28() {
  // 1. Get full-res image data to find bounding box
  const cw = canvas.width, ch = canvas.height;
  const fullData = ctx.getImageData(0, 0, cw, ch).data;

  // Find bounding box of drawn content
  let minX = cw, minY = ch, maxX = 0, maxY = 0;
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      if (fullData[(y * cw + x) * 4] > 20) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX <= minX || maxY <= minY) {
    return new Float32Array(784);
  }

  // 2. Crop and center into 20x20 area (MNIST uses 20x20 content + 4px padding)
  const bw = maxX - minX;
  const bh = maxY - minY;
  const side = Math.max(bw, bh);
  // Add margin so the digit doesn't fill edge-to-edge
  const margin = side * 0.15;
  const srcSide = side + margin * 2;

  // Center of bounding box
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;

  const tmp = document.createElement('canvas');
  tmp.width = 28; tmp.height = 28;
  const tctx = tmp.getContext('2d');
  tctx.fillStyle = '#000';
  tctx.fillRect(0, 0, 28, 28);

  // Draw into 20x20 center area (4px padding on each side, like MNIST)
  tctx.drawImage(canvas,
    cx - srcSide / 2, cy - srcSide / 2, srcSide, srcSide,
    4, 4, 20, 20
  );

  const imgData = tctx.getImageData(0, 0, 28, 28);
  const pixels = new Float32Array(784);
  for (let i = 0; i < 784; i++) {
    pixels[i] = imgData.data[i * 4] / 255.0;
  }
  return pixels;
}

function runInference() {
  if (!hasContent) return;
  const pixels = getPixels28();
  const result = infer(pixels, weights);
  updateDial(result.probs);
  updateInputPreview(result);
  updateNetworkDiagram(result);
}

// ===================== DIAL UPDATE =====================
function updateDial(probs) {
  const cells = document.querySelectorAll('.dial-cell[data-digit]');
  let maxProb = 0;
  for (let i = 0; i < 10; i++) if (probs[i] > maxProb) maxProb = probs[i];

  cells.forEach(cell => {
    const d = parseInt(cell.dataset.digit);
    const p = probs[d];
    const intensity = Math.min(p / Math.max(maxProb, 0.001), 1.0);

    // Color: cool(low) → warm(mid) → hot(high)
    let r, g, b;
    if (intensity < 0.5) {
      const t = intensity * 2;
      r = Math.round(30 + t * 200);
      g = Math.round(40 + t * 100);
      b = Math.round(80 - t * 40);
    } else {
      const t = (intensity - 0.5) * 2;
      r = Math.round(230 + t * 25);
      g = Math.round(140 - t * 100);
      b = Math.round(40 - t * 30);
    }

    const bgAlpha = 0.1 + intensity * 0.7;
    cell.style.background = `rgba(${r},${g},${b},${bgAlpha})`;
    cell.style.borderColor = intensity > 0.5
      ? `rgba(${r},${g},${b},0.8)`
      : 'var(--border)';
    cell.style.boxShadow = intensity > 0.6
      ? `0 0 ${intensity * 20}px rgba(${r},${g},${b},${intensity * 0.5})`
      : 'none';

    // Show percentage
    const pct = (p * 100).toFixed(0);
    cell.querySelector('span').textContent = d + '\\n' + pct + '%';
    cell.querySelector('span').style.whiteSpace = 'pre';
    cell.querySelector('span').style.fontSize = intensity > 0.5 ? '1rem' : '0.7rem';
    cell.querySelector('span').style.lineHeight = '1.3';
  });
}

// ===================== CNN INPUT PREVIEW =====================
function updateInputPreview(result) {
  const visInput = document.getElementById('vis-input');
  const ictx = visInput.getContext('2d');
  visInput.width = 28; visInput.height = 28;
  const iimg = ictx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = Math.round(result.input[i] * 255);
    iimg.data[i*4] = v; iimg.data[i*4+1] = v; iimg.data[i*4+2] = v; iimg.data[i*4+3] = 255;
  }
  ictx.putImageData(iimg, 0, 0);
}

function clearInputPreview() {
  const c = document.getElementById('vis-input');
  c.getContext('2d').clearRect(0, 0, c.width, c.height);
}

// ===================== NETWORK DIAGRAM =====================
const SVG_NS = 'http://www.w3.org/2000/svg';
const NN_W = 840, NN_H = 440;

// Layer definitions: name, node count, x position, radius
const LAYERS = [
  { name: 'Input',      count: 1,  x: 55,  r: 20 },
  { name: 'Conv1 (8)',  count: 8,  x: 210, r: 9 },
  { name: 'Conv2 (16)', count: 16, x: 380, r: 6 },
  { name: 'FC (64)',    count: 16, x: 580, r: 6 },   // 16 groups of 4 neurons
  { name: 'Output',     count: 10, x: 800, r: 8 },
];

function nodeY(layerIdx, nodeIdx) {
  const L = LAYERS[layerIdx];
  if (L.count === 1) return NN_H / 2;
  const margin = 30;
  return margin + (nodeIdx / (L.count - 1)) * (NN_H - 2 * margin);
}

// Pre-compute weight norms for link visualization
// Conv1→Conv2: wNorm12[i][j] = L2 norm of conv2_W[j,i,:,:]
const wNorm12 = [];
for (let i = 0; i < 8; i++) {
  wNorm12[i] = [];
  for (let j = 0; j < 16; j++) {
    let s = 0;
    for (let k = 0; k < 9; k++) { // 3x3 kernel
      const idx = (j * 8 + i) * 9 + k;
      s += weights.conv2_W.data[idx] * weights.conv2_W.data[idx];
    }
    wNorm12[i][j] = Math.sqrt(s);
  }
}

// Conv2→FC1 (grouped): wNorm2f[c][g] = sum abs weights from channel c (25 pixels) to group g (4 neurons)
const wNorm2f = [];
for (let c = 0; c < 16; c++) {
  wNorm2f[c] = [];
  for (let g = 0; g < 16; g++) {
    let s = 0;
    for (let px = 0; px < 25; px++) {
      for (let ni = 0; ni < 4; ni++) {
        const neuron = g * 4 + ni;
        const wi = (c * 25 + px) * 64 + neuron;
        s += Math.abs(weights.fc1_W[wi]);
      }
    }
    wNorm2f[c][g] = s;
  }
}

// FC1(grouped)→Output: wNormFO[g][o] = sum abs weights from group g (4 neurons) to output o
const wNormFO = [];
for (let g = 0; g < 16; g++) {
  wNormFO[g] = [];
  for (let o = 0; o < 10; o++) {
    let s = 0;
    for (let ni = 0; ni < 4; ni++) {
      const neuron = g * 4 + ni;
      s += Math.abs(weights.fc2_W[neuron * 10 + o]);
    }
    wNormFO[g][o] = s;
  }
}

// Normalize weight norms per layer-pair
function normalizeMatrix(mat) {
  let mx = 0;
  for (const row of mat) for (const v of row) if (v > mx) mx = v;
  if (mx > 0) for (const row of mat) for (let i = 0; i < row.length; i++) row[i] /= mx;
}
normalizeMatrix(wNorm12);
normalizeMatrix(wNorm2f);
normalizeMatrix(wNormFO);

// Build SVG elements
const svgLinks = document.getElementById('nn-links');
const svgNodes = document.getElementById('nn-nodes');
const svgLabels = document.getElementById('nn-labels');

// Store element references
const linkEls = {};  // linkEls['0-1'][srcIdx][dstIdx] = <line>
const nodeEls = [];  // nodeEls[layerIdx][nodeIdx] = <circle>

// Create links between adjacent layers
const LINK_PAIRS = [
  [0, 1], [1, 2], [2, 3], [3, 4]
];

for (const [li, lj] of LINK_PAIRS) {
  const key = li + '-' + lj;
  linkEls[key] = [];
  for (let si = 0; si < LAYERS[li].count; si++) {
    linkEls[key][si] = [];
    for (let dj = 0; dj < LAYERS[lj].count; dj++) {
      const line = document.createElementNS(SVG_NS, 'line');
      line.setAttribute('x1', LAYERS[li].x);
      line.setAttribute('y1', nodeY(li, si));
      line.setAttribute('x2', LAYERS[lj].x);
      line.setAttribute('y2', nodeY(lj, dj));
      line.setAttribute('stroke', '#ff6b35');
      line.setAttribute('stroke-width', '1');
      line.setAttribute('stroke-opacity', '0');
      svgLinks.appendChild(line);
      linkEls[key][si][dj] = line;
    }
  }
}

// Create nodes
for (let li = 0; li < LAYERS.length; li++) {
  nodeEls[li] = [];
  const L = LAYERS[li];
  for (let ni = 0; ni < L.count; ni++) {
    const circle = document.createElementNS(SVG_NS, 'circle');
    circle.setAttribute('cx', L.x);
    circle.setAttribute('cy', nodeY(li, ni));
    circle.setAttribute('r', L.r);
    circle.setAttribute('fill', '#30363d');
    circle.setAttribute('stroke', '#484f58');
    circle.setAttribute('stroke-width', '1');
    svgNodes.appendChild(circle);
    nodeEls[li][ni] = circle;

    // Output layer labels
    if (li === 4) {
      const text = document.createElementNS(SVG_NS, 'text');
      text.setAttribute('x', L.x + L.r + 6);
      text.setAttribute('y', nodeY(li, ni) + 4);
      text.setAttribute('text-anchor', 'start');
      text.setAttribute('font-size', '12');
      text.setAttribute('font-weight', '700');
      text.setAttribute('fill', '#e6edf3');
      text.textContent = ni;
      svgLabels.appendChild(text);
    }
  }
}

// Layer name labels at top
for (let li = 0; li < LAYERS.length; li++) {
  const text = document.createElementNS(SVG_NS, 'text');
  text.setAttribute('x', LAYERS[li].x);
  text.setAttribute('y', 12);
  text.setAttribute('text-anchor', 'middle');
  text.setAttribute('font-size', '13');
  text.setAttribute('fill', '#e6edf3');
  text.setAttribute('font-weight', '600');
  text.textContent = LAYERS[li].name;
  svgLabels.appendChild(text);
}

function updateNetworkDiagram(result) {
  // Compute per-node activations (average per channel/group)
  const act = [[], [], [], [], []];

  // Input: single node, average pixel brightness
  let inputSum = 0;
  for (let i = 0; i < 784; i++) inputSum += result.input[i];
  act[0][0] = Math.min(inputSum / 100, 1.0);

  // Conv1: 8 channels, average activation
  for (let ch = 0; ch < 8; ch++) {
    let s = 0;
    const off = ch * 24 * 24;
    for (let i = 0; i < 576; i++) s += result.conv1.data[off + i];
    act[1][ch] = s / 576;
  }

  // Conv2: 16 channels
  for (let ch = 0; ch < 16; ch++) {
    let s = 0;
    const off = ch * 10 * 10;
    for (let i = 0; i < 100; i++) s += result.conv2.data[off + i];
    act[2][ch] = s / 100;
  }

  // FC1: 16 groups of 4
  for (let g = 0; g < 16; g++) {
    let mx = 0;
    for (let ni = 0; ni < 4; ni++) {
      const v = result.fc1.data[g * 4 + ni];
      if (v > mx) mx = v;
    }
    act[3][g] = mx;
  }

  // Output: 10 probabilities
  for (let i = 0; i < 10; i++) act[4][i] = result.probs[i];

  // Normalize activations per layer to 0..1
  for (let li = 0; li < 5; li++) {
    let mx = 0;
    for (const v of act[li]) if (v > mx) mx = v;
    if (mx > 0) for (let i = 0; i < act[li].length; i++) act[li][i] /= mx;
  }

  // Update nodes
  for (let li = 0; li < 5; li++) {
    for (let ni = 0; ni < LAYERS[li].count; ni++) {
      const a = act[li][ni];
      const circle = nodeEls[li][ni];
      if (a > 0.5) {
        const t = (a - 0.5) * 2;
        const r = Math.round(255 * t);
        const g = Math.round(107 * t);
        const b = Math.round(53 * t);
        circle.setAttribute('fill', `rgb(${r},${g},${b})`);
        circle.setAttribute('stroke', `rgba(255,107,53,${a})`);
        circle.setAttribute('stroke-width', '2');
        if (a > 0.8) {
          circle.setAttribute('filter', 'url(#glow)');
        } else {
          circle.removeAttribute('filter');
        }
      } else {
        const t = a * 2;
        circle.setAttribute('fill', `rgba(88,166,255,${t * 0.3})`);
        circle.setAttribute('stroke', '#484f58');
        circle.setAttribute('stroke-width', '1');
        circle.removeAttribute('filter');
      }
    }
  }

  // Update links
  // Input → Conv1
  for (let j = 0; j < 8; j++) {
    const signal = act[0][0] * act[1][j];
    const el = linkEls['0-1'][0][j];
    el.setAttribute('stroke-opacity', (signal * 0.8).toFixed(3));
    el.setAttribute('stroke-width', signal > 0.3 ? '2' : '1');
  }

  // Conv1 → Conv2
  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 16; j++) {
      const signal = act[1][i] * act[2][j] * wNorm12[i][j];
      const el = linkEls['1-2'][i][j];
      el.setAttribute('stroke-opacity', Math.min(signal * 2, 0.8).toFixed(3));
      el.setAttribute('stroke-width', signal > 0.3 ? '1.5' : '0.5');
    }
  }

  // Conv2 → FC1
  for (let c = 0; c < 16; c++) {
    for (let g = 0; g < 16; g++) {
      const signal = act[2][c] * act[3][g] * wNorm2f[c][g];
      const el = linkEls['2-3'][c][g];
      el.setAttribute('stroke-opacity', Math.min(signal * 2, 0.8).toFixed(3));
      el.setAttribute('stroke-width', signal > 0.3 ? '1.5' : '0.5');
    }
  }

  // FC1 → Output
  for (let g = 0; g < 16; g++) {
    for (let o = 0; o < 10; o++) {
      const signal = act[3][g] * act[4][o] * wNormFO[g][o];
      const el = linkEls['3-4'][g][o];
      el.setAttribute('stroke-opacity', Math.min(signal * 3, 0.9).toFixed(3));
      el.setAttribute('stroke-width', signal > 0.2 ? '2' : '0.5');
    }
  }
}

function clearNetworkDiagram() {
  for (let li = 0; li < 5; li++) {
    for (let ni = 0; ni < LAYERS[li].count; ni++) {
      nodeEls[li][ni].setAttribute('fill', '#30363d');
      nodeEls[li][ni].setAttribute('stroke', '#484f58');
      nodeEls[li][ni].setAttribute('stroke-width', '1');
      nodeEls[li][ni].removeAttribute('filter');
    }
  }
  for (const key in linkEls) {
    for (const row of linkEls[key]) {
      for (const el of row) {
        el.setAttribute('stroke-opacity', '0');
      }
    }
  }
}

// ===================== MODEL INFO TOGGLE =====================
document.querySelector('.model-info').addEventListener('click', function(e) {
  if (e.target.classList.contains('copy-btn')) return;
  this.classList.toggle('open');
});
document.addEventListener('click', function(e) {
  const mi = document.querySelector('.model-info');
  if (!mi.contains(e.target)) mi.classList.remove('open');
});

// ===================== COPY MODEL INFO =====================
function copyModelInfo(e) {
  e.stopPropagation();
  const text = document.getElementById('model-tooltip').innerText.replace('コピー', '').trim();
  navigator.clipboard.writeText(text).then(() => {
    e.target.textContent = 'コピーしました';
    setTimeout(() => { e.target.textContent = 'コピー'; }, 1500);
  });
}

// ===================== INITIAL STATE =====================
updateDial(new Float32Array(10).fill(0.1));
"""


if __name__ == "__main__":
    # Quick test: generate with dummy weights
    import numpy as np
    dummy_weights = {
        "conv1_W": np.random.randn(8,1,5,5).tolist(),
        "conv1_b": np.zeros(8).tolist(),
        "conv2_W": np.random.randn(16,8,3,3).tolist(),
        "conv2_b": np.zeros(16).tolist(),
        "fc1_W": np.random.randn(400,64).tolist(),
        "fc1_b": np.zeros(64).tolist(),
        "fc2_W": np.random.randn(64,10).tolist(),
        "fc2_b": np.zeros(10).tolist(),
    }
    html = build_html(dummy_weights, {"loss":[],"acc":[],"val_acc":[]}, 0.0)
    print(f"Generated {len(html)} chars")
