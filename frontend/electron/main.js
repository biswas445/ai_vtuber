/**
 * Electron main process: a borderless, transparent, always-on-top overlay
 * window (airi-vtuber style) that hosts the VRM companion.
 *
 * - The window is click-through by default; the renderer hit-tests the character
 *   and toggles interactivity over IPC, so clicks on empty space fall through
 *   to whatever app is underneath.
 * - Model files are served over a custom `vtuber-model://` protocol so the
 *   renderer can fetch them in both dev and packaged builds.
 */

const {
  app,
  BrowserWindow,
  ipcMain,
  screen,
  protocol,
  net,
  Tray,
  Menu,
  nativeImage,
} = require('electron');
const { spawn } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');
const { pathToFileURL } = require('node:url');

// TTS autoplay: the overlay window rarely holds keyboard focus, and Chromium
// would otherwise refuse <audio>.play() without a user gesture — which would
// silently kill every voice reply coming from the backend.
app.commandLine.appendSwitch('autoplay-policy', 'no-user-gesture-required');

const MODELS_DIR = path.join(__dirname, '..', 'characters_models');
const MODEL_SCHEME = 'vtuber-model';
const WINDOW_W = 630;
const WINDOW_H = 840;
// The renderer's fit math cannot work below this (it skips degenerate
// viewports); a floor keeps the window from being parked there. The edit
// mode's resize handles enforce the same minimums in the renderer.
const MIN_WIN_W = 160;
const MIN_WIN_H = 200;

let win = null;
let tray = null;
let dragOffset = null;

const state = {
  x: null,
  y: null,
  w: WINDOW_W,
  h: WINDOW_H,
  pinned: true,
  trackMouse: true, // stream the global cursor position to the renderer
  voice: true, // TTS replies play through the overlay (with lip-sync)
  model: null, // rel path of the active .vrm file
  // Character offset inside the window, fractions of the window size
  // (survives window resizes). The window itself is the container the
  // character is fitted into — resizing it re-fits her live.
  charOffset: { x: 0, y: 0 },
};

// Edit mode is session-only: always start with the overlay locked.
let editMode = 'off'; // 'off' | 'container' | 'character'

/* ------------------------------------------------------------------ state */

function stateFile() {
  return path.join(app.getPath('userData'), 'overlay-state.json');
}

function loadState() {
  try {
    const saved = JSON.parse(fs.readFileSync(stateFile(), 'utf8'));
    // Old versions stored a character container INSIDE the overlay window;
    // the window itself is the container now. Fold the old container into
    // the character offset so she keeps her spot, then drop it. (The 0.98
    // mirrors the renderer's bottom fit margin, which scaled with the
    // container's height.)
    const c = saved.container;
    if (c && typeof c === 'object') {
      const cx = Number(c.x) || 0;
      const cy = Number(c.y) || 0;
      const cw = Number.isFinite(Number(c.w)) ? Number(c.w) : 1;
      const ch = Number.isFinite(Number(c.h)) ? Number(c.h) : 1;
      if (cx !== 0 || cy !== 0 || cw !== 1 || ch !== 1) {
        const off = saved.charOffset || state.charOffset;
        saved.charOffset = {
          x: (Number(off.x) || 0) + cx + (cw - 1) / 2,
          y: (Number(off.y) || 0) + cy + (ch - 1) * 0.98,
        };
      }
    }
    delete saved.container;
    Object.assign(state, saved);
  } catch {
    /* first run */
  }
}

let saveTimer = null;
function writeState() {
  try {
    fs.mkdirSync(path.dirname(stateFile()), { recursive: true });
    fs.writeFileSync(stateFile(), JSON.stringify(state, null, 2));
  } catch (err) {
    console.error('[state] failed to save:', err.message);
  }
}
function saveState() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    saveTimer = null;
    writeState();
  }, 250);
}

/* ----------------------------------------------------------------- models */

function scanModels() {
  const found = [];
  const walk = (dir) => {
    let entries;
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(full);
      } else if (entry.name.toLowerCase().endsWith('.vrm')) {
        found.push({
          name: entry.name.replace(/\.vrm$/i, ''),
          rel: path.relative(MODELS_DIR, full).split(path.sep).join('/'),
        });
      }
    }
  };
  walk(MODELS_DIR);
  found.sort((a, b) => a.name.localeCompare(b.name));
  return found;
}

function modelUrl(rel) {
  return `${MODEL_SCHEME}://local/${rel.split('/').map(encodeURIComponent).join('/')}`;
}

/* ------------------------------------------------------------------ tray */

function createTrayIcon() {
  // 16x16 pink dot, drawn as raw BGRA. createFromBitmap is Windows-only,
  // which is the only platform this overlay targets for now.
  const size = 16;
  const buf = Buffer.alloc(size * size * 4);
  const c = (size - 1) / 2;
  const r = 6.5;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const d = Math.hypot(x - c, y - c);
      if (d > r) continue;
      const i = (y * size + x) * 4;
      buf[i] = 0x9d; // B
      buf[i + 1] = 0x6b; // G
      buf[i + 2] = 0xff; // R
      buf[i + 3] = d > r - 1 ? Math.round(255 * (r - d)) : 255;
    }
  }
  if (typeof nativeImage.createFromBitmap === 'function') {
    return nativeImage.createFromBitmap(buf, { width: size, height: size });
  }
  return nativeImage.createEmpty();
}

function buildTrayMenu() {
  const models = scanModels();
  return Menu.buildFromTemplate([
    {
      label: 'Pinned on top',
      type: 'checkbox',
      checked: state.pinned,
      click: (item) => setPinned(item.checked),
    },
    {
      label: 'Track mouse',
      type: 'checkbox',
      checked: state.trackMouse,
      click: (item) => setTrackMouse(item.checked),
    },
    {
      label: 'Voice replies (TTS)',
      type: 'checkbox',
      checked: state.voice !== false,
      click: (item) => setVoice(item.checked),
    },
    {
      label: 'Resize window',
      type: 'checkbox',
      checked: editMode === 'container',
      click: (item) => setEditMode(item.checked ? 'container' : 'off'),
    },
    {
      label: 'Move character',
      type: 'checkbox',
      checked: editMode === 'character',
      click: (item) => setEditMode(item.checked ? 'character' : 'off'),
    },
    { type: 'separator' },
    ...models.map((m) => ({
      label: m.name,
      type: 'radio',
      checked: state.model === m.rel,
      click: () => setModel(m.rel),
    })),
    { type: 'separator' },
    {
      label: 'Restart voice engine',
      click: () => {
        stopBackend();
        startBackend();
      },
    },
    { label: 'Quit', click: () => app.quit() },
  ]);
}

function createTray() {
  tray = new Tray(createTrayIcon());
  tray.setToolTip('AI companion overlay');
  tray.setContextMenu(buildTrayMenu());
}

function refreshTray() {
  if (tray) tray.setContextMenu(buildTrayMenu());
}

/* ----------------------------------------------------------------- window */

// Minimum px of the window that must stay on-screen on each axis. Like any
// normal app window, the overlay may be dragged so it hangs off ANY screen
// edge — half of it (or more) off-screen is fine — but it can never be
// pushed FULLY off-screen: a visible, grabbable portion always remains so it
// can't be lost. `getDisplayNearestPoint` also handles a position saved on a
// monitor that is no longer attached by snapping to the nearest real display.
// The size defaults to the boot-time constants but callers that know the
// LIVE window size must pass it — the window is resizable.
const MIN_WIN_VISIBLE_PX = 100;
function clampToDisplay(x, y, w = WINDOW_W, h = WINDOW_H) {
  const area = screen.getDisplayNearestPoint({ x, y }).workArea;
  const keepX = Math.min(MIN_WIN_VISIBLE_PX, w);
  const keepY = Math.min(MIN_WIN_VISIBLE_PX, h);
  return {
    // Left/right: at least keepX px of the window stay inside the display.
    x: Math.min(Math.max(x, area.x - (w - keepX)), area.x + area.width - keepX),
    // Top/bottom: at least keepY px of the window stay inside the display.
    y: Math.min(Math.max(y, area.y - (h - keepY)), area.y + area.height - keepY),
  };
}

function setPinned(pinned) {
  state.pinned = pinned;
  if (win) {
    win.setAlwaysOnTop(pinned, pinned ? 'screen-saver' : 'normal');
  }
  refreshTray();
  saveState();
  broadcastState();
}

function setTrackMouse(enabled) {
  state.trackMouse = enabled;
  refreshTray();
  saveState();
  broadcastState();
}

function setVoice(enabled) {
  state.voice = enabled;
  refreshTray();
  saveState();
  sendBackendConfig(); // the pipeline skips synthesis while muted
  broadcastState();
}

function setEditMode(mode) {
  editMode = mode === 'container' || mode === 'character' ? mode : 'off';
  refreshTray();
  broadcastState();
}

function setModel(rel) {
  state.model = rel;
  refreshTray();
  saveState();
  if (win) {
    win.webContents.send('overlay:model-changed', { rel, url: modelUrl(rel) });
  }
}

function broadcastState() {
  if (win) {
    win.webContents.send('overlay:state-changed', {
      pinned: state.pinned,
      trackMouse: state.trackMouse,
      voice: state.voice,
      model: state.model,
      editMode,
    });
  }
}

function createWindow() {
  const display = screen.getPrimaryDisplay();
  const { width: sw, height: sh } = display.workArea;

  // Restore the size the user last gave her (edit-mode resize or native
  // edge drag); guard against a hand-edited/corrupt state file.
  const w = Number.isFinite(Number(state.w))
    ? Math.max(MIN_WIN_W, Math.round(state.w))
    : WINDOW_W;
  const h = Number.isFinite(Number(state.h))
    ? Math.max(MIN_WIN_H, Math.round(state.h))
    : WINDOW_H;

  let x;
  let y;
  if (typeof state.x === 'number' && typeof state.y === 'number') {
    ({ x, y } = clampToDisplay(state.x, state.y, w, h));
  } else {
    x = Math.round(display.workArea.x + sw - w - 40);
    y = Math.round(display.workArea.y + sh - h - 20);
  }

  win = new BrowserWindow({
    width: w,
    height: h,
    minWidth: MIN_WIN_W,
    minHeight: MIN_WIN_H,
    x,
    y,
    frame: false,
    transparent: true,
    backgroundColor: '#00000000',
    hasShadow: false,
    // Resizable/maximizable so the companion can be scaled to taste; the
    // renderer re-fits her position on every size change so she stays centered
    // and never clips. Kept non-minimizable: with `skipTaskbar` a minimized
    // overlay would have no way to be restored.
    resizable: true,
    maximizable: true,
    minimizable: false,
    fullscreenable: false,
    skipTaskbar: true,
    alwaysOnTop: state.pinned,
    roundedCorners: false,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      backgroundThrottling: false,
    },
  });

  if (state.pinned) win.setAlwaysOnTop(true, 'screen-saver');
  win.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
  // Start click-through; the renderer re-enables input when the pointer is
  // over the character or the controls.
  win.setIgnoreMouseEvents(true, { forward: true });

  win.once('ready-to-show', () => win.show());

  win.on('moved', () => {
    const [wx, wy] = win.getPosition();
    state.x = wx;
    state.y = wy;
    saveState();
  });

  win.on('resized', () => {
    // A maximize fires this too — persist the user's real size, which the
    // un-maximize resize delivers, not the maximized one.
    if (win.isMaximized()) return;
    const [ww, wh] = win.getSize();
    state.w = ww;
    state.h = wh;
    saveState();
  });

  win.on('closed', () => {
    win = null;
  });

  const devUrl = process.env.VITE_DEV_SERVER_URL;
  if (devUrl) {
    win.loadURL(devUrl);
  } else {
    win.loadFile(path.join(__dirname, '..', 'dist', 'renderer', 'index.html'));
  }

  // Forward renderer console output here so `npm run dev` shows everything.
  // Single-argument form: Electron picks the legacy (level, message, ...)
  // signature when the listener declares extra parameters, which is deprecated.
  win.webContents.on('console-message', (event) => {
    console.log(`[renderer:${event.level}] ${event.message}`);
  });
  win.webContents.on('preload-error', (_e, preloadPath, error) => {
    console.error('[preload-error]', preloadPath, error);
  });
  win.webContents.on('did-fail-load', (_e, code, desc) => {
    console.error('[did-fail-load]', code, desc);
  });
}

/* -------------------------------------------------------- cursor tracking */

// The renderer only receives mousemove while the pointer is inside the window,
// so the companion's gaze would stop at the window's edge. Poll the global
// cursor position and forward it window-relative so she can follow the pointer
// anywhere on screen. Only forwarded when the cursor actually moves — and not
// polled at all while mouse tracking is turned off.
function startCursorTracking() {
  let last = null;
  setInterval(() => {
    if (!win || win.isDestroyed()) return;
    if (!state.trackMouse) {
      last = null; // forget the old spot so re-enabling picks up fresh moves
      return;
    }
    const point = screen.getCursorScreenPoint();
    if (last && point.x === last.x && point.y === last.y) return;
    last = point;
    const [wx, wy] = win.getPosition();
    win.webContents.send('overlay:cursor', { x: point.x - wx, y: point.y - wy });
  }, 30);
}

/* ------------------------------------------------------- voice backend */

// assistant.py runs as a child process — the full voice pipeline
// (mic -> VAD -> STT -> LLM -> TTS). In BRIDGE mode its stdout is a
// JSON-lines protocol ({type, value} events: state changes and TTS clips as
// data: URIs) while all of its logging goes to stderr. Config (voice on/off)
// is written to its stdin. Events are forwarded to the renderer as
// 'backend:event' IPC, where the character lip-syncs the clips and shows
// the listening/thinking/speaking states.

const PROJECT_ROOT = path.join(__dirname, '..', '..');
const BACKEND_SCRIPT = path.join(PROJECT_ROOT, 'assistant.py');

let backend = null;
let backendBuf = '';
let backendWanted = false;
// Backend events emitted before the renderer reports 'overlay:ready' are
// buffered here and flushed once she's up: sent earlier, they'd either be
// dropped by a renderer that hasn't registered its listener yet, or hit
// `speak()` before the model exists and be acknowledged without ever
// playing (the greeting would vanish on a slow model load).
let rendererReady = false;
let pendingEvents = [];
const PENDING_EVENT_LIMIT = 128;

function sendBackendEvent(event) {
  if (!rendererReady) {
    pendingEvents.push(event);
    // Pathological never-ready renderer: keep the newest events, ack the
    // dropped clips so the backend never stalls on their confirmations
    // (its own grace timeout would also save it — this just avoids the wait).
    if (pendingEvents.length > PENDING_EVENT_LIMIT) {
      const dropped = pendingEvents.splice(0, pendingEvents.length - PENDING_EVENT_LIMIT);
      for (const ev of dropped) {
        if (ev.type !== 'speak') continue;
        const id = ev.value && typeof ev.value === 'object' ? ev.value.id : null;
        if (backend && backend.stdin && !backend.stdin.destroyed) {
          backend.stdin.write(JSON.stringify({ type: 'tts-done', id }) + '\n');
        }
      }
    }
    return;
  }
  if (win && !win.isDestroyed()) win.webContents.send('backend:event', event);
}

function pythonExe() {
  const venvPy = path.join(PROJECT_ROOT, 'venv', 'Scripts', 'python.exe');
  try {
    if (fs.existsSync(venvPy)) return venvPy;
  } catch {
    /* fall through to PATH */
  }
  return 'python';
}

function startBackend() {
  if (backend || backendWanted) return;
  if (!fs.existsSync(BACKEND_SCRIPT)) {
    console.warn('[backend] assistant.py not found — voice pipeline disabled');
    return;
  }
  backendWanted = true;
  const child = spawn(pythonExe(), [BACKEND_SCRIPT], {
    cwd: PROJECT_ROOT,
    env: {
      ...process.env,
      BRIDGE: '1',
      PYTHONIOENCODING: 'utf-8',
      PYTHONUNBUFFERED: '1',
    },
    windowsHide: true,
  });
  backend = child;
  console.log('[backend] starting voice pipeline (model load can take a while)...');

  // A write racing the backend's death would otherwise emit an unhandled
  // 'error' (EPIPE) on the stdin socket and take down the whole main
  // process; the exit handler below does the actual cleanup.
  child.stdin.on('error', () => {});

  child.stdout.setEncoding('utf8');
  child.stdout.on('data', (chunk) => {
    backendBuf += chunk;
    let nl;
    while ((nl = backendBuf.indexOf('\n')) >= 0) {
      const line = backendBuf.slice(0, nl).trim();
      backendBuf = backendBuf.slice(nl + 1);
      if (!line) continue;
      let event = null;
      try {
        event = JSON.parse(line);
      } catch {
        console.log(`[backend] ${line}`); // not protocol — pass through as log
        continue;
      }
      if (!event || typeof event.type !== 'string') continue;
      if (event.type === 'speak' && !state.voice) {
        // Muted: the clip is never played, so nobody will report it done —
        // answer the backend ourselves (echoing the clip id so it matches
        // the wait) or it would stall on the timeout.
        if (backend && backend.stdin && !backend.stdin.destroyed) {
          const id =
            event.value && typeof event.value === 'object' ? event.value.id : null;
          backend.stdin.write(JSON.stringify({ type: 'tts-done', id }) + '\n');
        }
        continue;
      }
      sendBackendEvent(event);
    }
  });

  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (chunk) => process.stdout.write(`[backend] ${chunk}`));

  child.on('error', (err) => {
    console.error('[backend] failed to start:', err.message);
    if (backend === child) {
      backend = null;
      backendWanted = false;
    }
  });

  child.on('exit', (code) => {
    // A tray restart spawns the replacement BEFORE this event fires. Only
    // the current backend may clear the slot — otherwise the exit of the
    // old child orphans the new one: config/tts-done writes never reach
    // its stdin, it survives app exit still holding the mic, and every
    // further restart stacks another orphan forwarding duplicate events.
    if (backend !== child) return;
    backend = null;
    backendWanted = false;
    console.log(`[backend] voice pipeline exited (code ${code})`);
    sendBackendEvent({ type: 'state', value: 'idle' });
  });

  sendBackendConfig();
}

function stopBackend() {
  if (!backend) return;
  const child = backend;
  backend = null;
  backendWanted = false;
  try {
    child.kill();
  } catch {
    /* already gone */
  }
}

function sendBackendConfig() {
  if (backend && backend.stdin && !backend.stdin.destroyed) {
    backend.stdin.write(JSON.stringify({ type: 'config', voice: state.voice }) + '\n');
  }
}

/* -------------------------------------------------------------------- ipc */

function registerIpc() {
  ipcMain.on('overlay:set-ignore', (_e, ignore) => {
    if (win) win.setIgnoreMouseEvents(Boolean(ignore), { forward: true });
  });

  ipcMain.on('overlay:drag-start', (_e, grabX, grabY) => {
    dragOffset = { x: Number(grabX) || 0, y: Number(grabY) || 0 };
  });

  ipcMain.on('overlay:drag-move', (_e, screenX, screenY) => {
    if (!win || !dragOffset) return;
    // Clamped with the same limits as boot (and the window's LIVE size), so
    // the window can never be dragged into a spot where it (and the
    // character) vanishes off-screen.
    const [ww, wh] = win.getSize();
    const pos = clampToDisplay(
      Math.round(screenX - dragOffset.x),
      Math.round(screenY - dragOffset.y),
      ww,
      wh,
    );
    win.setPosition(pos.x, pos.y);
  });

  ipcMain.on('overlay:drag-end', () => {
    dragOffset = null;
    if (win) {
      const [wx, wy] = win.getPosition();
      state.x = wx;
      state.y = wy;
      saveState();
    }
  });

  ipcMain.handle('overlay:toggle-pin', () => {
    setPinned(!state.pinned);
    return state.pinned;
  });

  ipcMain.handle('overlay:toggle-track-mouse', () => {
    setTrackMouse(!state.trackMouse);
    return state.trackMouse;
  });

  ipcMain.handle('overlay:toggle-voice', () => {
    setVoice(!state.voice);
    return state.voice;
  });

  // The renderer signals boot-complete; only then does the voice backend
  // start, so its state/speak events cannot be lost. Any events the backend
  // already produced (e.g. the 12 s fallback fired during a slow model
  // load) are flushed to the renderer now that it can actually handle them.
  // A fallback timer below covers a renderer that never reports ready.
  ipcMain.on('overlay:ready', () => {
    rendererReady = true;
    for (const event of pendingEvents.splice(0)) {
      win?.webContents?.send?.('backend:event', event);
    }
    startBackend();
  });

  // The renderer reports that the TTS clip has FINISHED PLAYING. Forwarded
  // to the backend's stdin — it keeps the mic hard-gated until this lands,
  // so she can never hear her own voice. The id echoes the clip the ack
  // belongs to; the backend ignores confirmations for any other clip (stale
  // acks, or acks crossing a backend restart).
  ipcMain.on('overlay:tts-done', (_e, id) => {
    if (backend && backend.stdin && !backend.stdin.destroyed) {
      backend.stdin.write(
        JSON.stringify({ type: 'tts-done', id: id == null ? null : id }) + '\n',
      );
    }
  });

  ipcMain.handle('overlay:get-state', () => {
    const models = scanModels();
    // The saved model may no longer exist (e.g. models were replaced) — fall
    // back to the first available one instead of loading nothing.
    if (!models.some((m) => m.rel === state.model)) {
      state.model = models.length ? models[0].rel : null;
      refreshTray();
      saveState();
    }
    return {
      pinned: state.pinned,
      trackMouse: state.trackMouse,
      voice: state.voice,
      model: state.model,
      modelUrl: state.model ? modelUrl(state.model) : null,
      models,
      charOffset: state.charOffset,
      editMode,
    };
  });

  ipcMain.handle('overlay:set-model', (_e, rel) => {
    setModel(rel);
    return { rel, url: modelUrl(rel) };
  });

  ipcMain.handle('overlay:set-edit-mode', (_e, mode) => {
    setEditMode(mode);
    return editMode;
  });

  // Edit-mode resize/move: the renderer sends the desired window bounds in
  // SCREEN coordinates (the window moves/resizes under the pointer mid-drag,
  // so window-relative coords can't describe it). Size is floored at the
  // window minimums and the position clamped with the same limits as boot
  // and the window drag, so she can never be lost off-screen. The 'moved'/
  // 'resized' handlers persist the result.
  ipcMain.on('overlay:set-window-bounds', (_e, bounds) => {
    if (!win || win.isDestroyed() || !bounds || typeof bounds !== 'object') return;
    const w = Math.max(MIN_WIN_W, Math.round(Number(bounds.w) || MIN_WIN_W));
    const h = Math.max(MIN_WIN_H, Math.round(Number(bounds.h) || MIN_WIN_H));
    const [curX, curY] = win.getPosition();
    const rawX = Number(bounds.x);
    const rawY = Number(bounds.y);
    const pos = clampToDisplay(
      Number.isFinite(rawX) ? Math.round(rawX) : curX,
      Number.isFinite(rawY) ? Math.round(rawY) : curY,
      w,
      h,
    );
    win.setBounds({ x: pos.x, y: pos.y, width: w, height: h });
  });

  // The character offset arrives as fractions of the window. The renderer
  // already clamps it; this is just belt-and-braces against a malformed
  // payload corrupting the saved state. It may legitimately exceed ±1: the
  // character is allowed to hang off the window edges (a visible anchor
  // always stays in view).
  const clampNum = (v, lo, hi, fallback) => {
    const n = Number(v);
    return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : fallback;
  };
  ipcMain.on('overlay:set-char-offset', (_e, charOffset) => {
    if (!charOffset || typeof charOffset !== 'object') return;
    state.charOffset = {
      // Wide range: with the scale multiplier the model can dwarf the
      // window, pushing its clamped offset well beyond one window width.
      x: clampNum(charOffset.x, -4, 4, state.charOffset.x),
      y: clampNum(charOffset.y, -4, 4, state.charOffset.y),
    };
    saveState();
  });

  ipcMain.on('overlay:close', () => app.quit());
}

/* ------------------------------------------------------------------ boot */

// Serve character model files to the renderer over a privileged scheme.
protocol.registerSchemesAsPrivileged([
  {
    scheme: MODEL_SCHEME,
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      stream: true,
      corsEnabled: true,
    },
  },
]);

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (win) win.show();
  });

  app.whenReady().then(() => {
    app.setAppUserModelId('ai-assistant.companion-overlay');

    protocol.handle(MODEL_SCHEME, async (request) => {
      try {
        const url = new URL(request.url);
        const rel = decodeURIComponent(url.pathname).replace(/^\/+/, '');
        const target = path.resolve(MODELS_DIR, rel);
        if (
          target !== MODELS_DIR &&
          !target.startsWith(MODELS_DIR + path.sep)
        ) {
          return new Response('Forbidden', { status: 403 });
        }
        if (!fs.existsSync(target) || !fs.statSync(target).isFile()) {
          return new Response('Not found', { status: 404 });
        }
        const res = await net.fetch(pathToFileURL(target).toString());
        // three.js loads textures through <img crossorigin>, so the
        // responses must carry an explicit CORS allowance.
        const headers = new Headers(res.headers);
        headers.set('Access-Control-Allow-Origin', '*');
        return new Response(res.body, {
          status: res.status,
          statusText: res.statusText,
          headers,
        });
      } catch (err) {
        console.error('[model-protocol]', err);
        return new Response('Bad request', { status: 400 });
      }
    });

    loadState();
    registerIpc();
    createWindow();
    createTray();
    startCursorTracking();
    // Fallback in case the renderer never reports ready (startBackend is
    // guarded, so a later 'overlay:ready' cannot double-start it; events it
    // produces early are buffered until the renderer is up).
    setTimeout(() => startBackend(), 12000);
  });

  app.on('window-all-closed', () => app.quit());

  // Flush a pending (debounced) state save so the last window position /
  // model choice is not lost when the app quits within the debounce window,
  // and take the voice pipeline down with the app.
  app.on('before-quit', () => {
    stopBackend();
    if (saveTimer !== null) {
      clearTimeout(saveTimer);
      saveTimer = null;
      writeState();
    }
  });
}
