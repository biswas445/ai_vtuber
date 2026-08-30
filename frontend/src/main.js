import './styles.css';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import {
  createIcons,
  Pin,
  PinOff,
  RefreshCw,
  Plus,
  Minus,
  X,
  MousePointer,
  MousePointer2Off,
  ScanBox,
  PersonStanding,
  Volume2,
  VolumeX,
} from 'lucide';
import { IdleDriver } from './idle.js';
import { setupOverlay } from './overlay.js';
import { setupEditMode } from './edit-mode.js';
import {
  parseWavDataUri,
  buildVisemeTimeline,
  sampleTimeline,
  VOWEL_CENTERS,
  VOWEL_OPEN,
} from './lipsync.mjs';

createIcons({
  icons: {
    Pin,
    PinOff,
    RefreshCw,
    Plus,
    Minus,
    X,
    MousePointer,
    MousePointer2Off,
    ScanBox,
    PersonStanding,
    Volume2,
    VolumeX,
  },
});

/* ---------------------------------------------------------------- stage --- */
// Transparent WebGL canvas hosted by the frameless overlay window. The
// character is rendered against a fully transparent clear color so only she
// (and the toolbar) is visible on the desktop.

const CAMERA_FOV = 28; // narrow, portrait-lens feel — less perspective warp
const CAMERA_DIST = 3; // model plane sits at z=0, camera looks down -Z

const renderer = new THREE.WebGLRenderer({
  alpha: true,
  antialias: true,
  powerPreference: 'high-performance',
});
renderer.setClearColor(0x000000, 0);
// Render at the display's full native resolution — the overlay window is
// small, so even high-DPI screens stay cheap, and the character stays crisp.
renderer.setPixelRatio(window.devicePixelRatio || 1);
document.getElementById('stage-holder').appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(CAMERA_FOV, 1, 0.05, 60);
camera.position.set(0, 0, CAMERA_DIST);

// Soft, even lighting suits the toon-shaded MToon materials: a hemisphere
// fill plus a key light from in front and slightly above so the face is
// always lit regardless of the model's own light setup. A cool rim light
// from behind separates her outline from whatever is on the desktop.
scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa4b2, 1.6));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.7);
keyLight.position.set(0.6, 1.8, 2.2);
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0xdfe8ff, 0.8);
rimLight.position.set(-1.4, 1.4, -2.2);
scene.add(rimLight);

const state = {
  models: [],
  current: null, // rel path of the loaded model
  vrm: null, // VRM instance
  rig: null, // Group carrying the fitted position/scale
  body: null, // Group pivoted at her feet — the idle driver poses this
  unitBounds: null, // model bounds at unit scale (measured once per load)
  idle: null, // IdleDriver
  scaleMul: 1,
  trackMouse: true, // whether the gaze follows the global cursor
  charRect: null, // screen-space character rect for hit-testing (per frame)
  // User's character offset inside the window, fractions of the window
  // (x+ = right, y+ = down on screen). Default: centered, feet on the
  // window's bottom margin. The window itself is the container the
  // character is fitted into — resizing it re-fits her live.
  charOffset: { x: 0, y: 0 },
  editMode: 'off', // 'off' | 'container' | 'character'
};

/* ---------------------------------------------------------------- loader --- */

const loader = new GLTFLoader();
// The plugin turns a glTF load into a VRM load: gltf.userData.vrm carries the
// VRM instance (humanoid, expressions, lookAt, spring bones) for either the
// VRM 0.x or VRM 1.0 format.
loader.register((parser) => new VRMLoaderPlugin(parser));

function modelUrlFor(rel) {
  return `vtuber-model://local/${rel.split('/').map(encodeURIComponent).join('/')}`;
}

/* ------------------------------------------------------------------- fit --- */
// The camera is fixed; fitting works in screen space: at the model plane
// (z=0) one world unit subtends a known number of pixels, so the model is
// scaled and placed so her feet rest just above the bottom safe margin,
// centered, head never clipped.

// Safe margins (fractions of the viewport) kept around the character when
// fitting. The top margin is generous so upward moves (hop rises, head
// motion) have headroom and the head never clips the top edge; the bottom
// pad keeps the feet just above the window edge.
const FIT_MARGIN_SIDE = 0.04;
const FIT_MARGIN_TOP = 0.1;
const FIT_MARGIN_BOTTOM = 0.02;
const FIT_BOTTOM_PAD = 4;

// Last viewport the fit was computed for — the render loop compares these to
// the live size every frame so a missed/stale resize event (minimize →
// maximize, DPI changes) can never leave the character mis-positioned.
let lastFitW = 0;
let lastFitH = 0;
let lastFitDPR = 0;
// Fitted geometry from the last fitModel run — shared with the character
// drag so it clamps with exactly the same limits the rendering uses.
let lastFitGeometry = null; // { widthPx, heightPx, anchorX, anchorY }

function pxPerUnit() {
  const visibleHeight = 2 * CAMERA_DIST * Math.tan(THREE.MathUtils.degToRad(CAMERA_FOV / 2));
  return window.innerHeight / visibleHeight;
}

/** Map window px onto the model plane (world coords). The gaze target lives a
 * little in front of the model so the eyes converge naturally toward it. */
function gazeToWorld(xPx, yPx, out) {
  const ppu = pxPerUnit();
  return out.set(
    (xPx - window.innerWidth / 2) / ppu,
    (window.innerHeight / 2 - yPx) / ppu,
    0.4,
  );
}

/**
 * Measure the model's bounds at unit scale (once) and set its resting
 * transform. VRM bounds are available as soon as the file has parsed (no
 * first-render wait), but keep the retry path in case a model ever isn't.
 */
function ensureUnitBounds() {
  const vrm = state.vrm;
  if (!vrm) return false;
  if (state.unitBounds) return true;

  // Measure with a clean transform so we capture the artwork's true extent.
  vrm.scene.updateWorldMatrix(true, true);
  const box = new THREE.Box3().setFromObject(vrm.scene);
  if (box.isEmpty()) return false;

  state.unitBounds = {
    minX: box.min.x,
    minY: box.min.y,
    width: box.max.x - box.min.x,
    height: box.max.y - box.min.y,
    cx: (box.min.x + box.max.x) / 2,
  };
  // Keep the cache entry and the idle driver's amplitude scale in sync with
  // the late measurement.
  const entry = modelCache.get(state.current);
  if (entry) entry.unitBounds = state.unitBounds;
  if (state.idle) {
    state.idle.fit.unitSize.width = state.unitBounds.width;
    state.idle.fit.unitSize.height = state.unitBounds.height;
  }
  applyRestTransform();
  return true;
}

/**
 * Offset the model inside the body group so her feet's bottom-center sits at
 * the group's origin — whole-body poses (sways, rises, leans) then pivot
 * around the feet.
 */
function applyRestTransform() {
  const vrm = state.vrm;
  const b = state.unitBounds;
  if (!vrm || !b) return;
  vrm.scene.position.set(-b.cx, -b.minY, 0);
}

/** The container rect in window px — the window itself is the container. */
function containerPx() {
  return { x: 0, y: 0, w: window.innerWidth, h: window.innerHeight };
}

// Minimum px of the model that must stay inside the window on each axis.
// She may hang off ANY edge (the overlay clips her), but can never be pushed
// entirely off-screen — a visible, grabbable portion always remains.
const MIN_VISIBLE_PX = 80;

/**
 * Clamp a feet anchor (window px) so she can never be lost off-screen — the
 * ONE limit used by both the rendering (fitModel) and the character drag,
 * which is what makes the drag symmetric: the character follows the cursor
 * exactly everywhere she can actually be rendered, and stops precisely
 * where the render clamp stops her. The model may cross any window edge —
 * left, right, top and bottom alike — as long as at least MIN_VISIBLE_PX px
 * of it stay in view on that axis.
 */
function clampFeetPx(feetPx, feetPy, widthPx, heightPx, W, H) {
  const half = widthPx / 2;
  const keepX = Math.min(MIN_VISIBLE_PX, widthPx);
  const keepY = Math.min(MIN_VISIBLE_PX, heightPx);
  // Horizontal: at least keepX px of the model stay inside the window.
  const x = Math.min(Math.max(feetPx, keepX - half), W - keepX + half);
  // feetPy is the model's BOTTOM; its top is feetPy - heightPx. Going up the
  // bottom must stay in view; going down the top must stay in view.
  const y = Math.min(Math.max(feetPy, keepY), H - keepY + heightPx);
  return { x, y };
}

/**
 * Clamp a proposed character offset (window px) with the same visibility
 * limits the rendering applies (see clampFeetPx). Used live during the
 * character drag.
 */
function clampCharOffsetPx(offPx) {
  const g = lastFitGeometry;
  if (!g) return offPx;
  const feet = clampFeetPx(
    g.anchorX + offPx.x,
    g.anchorY + offPx.y,
    g.widthPx,
    g.heightPx,
    window.innerWidth,
    window.innerHeight,
  );
  return { x: feet.x - g.anchorX, y: feet.y - g.anchorY };
}

/**
 * Fit the character to the current viewport: canvas size, camera aspect,
 * model scale and position are all recomputed from scratch, so a
 * resize/maximize/restore can never accumulate drift or clip the face.
 * The window itself is the container: she is fitted into the full window,
 * then shifted by the user's character offset.
 *
 * Positioning works in window PX (intuitive, resolution-independent math),
 * then converts to world once at the end:
 *  1. feet anchor = window bottom safe margin + user offset,
 *  2. clamp the resulting model rect against the window: she may hang off
 *     any edge, but at least MIN_VISIBLE_PX px of her stay in view, so she
 *     can never be lost off-screen. Offsets are stored as fractions of the
 *     window, which a maximize can amplify — this clamp is what keeps her
 *     visible regardless. The clamp result is written back to
 *     state.charOffset so stored, rendered and persisted positions agree.
 */
function fitModel() {
  const vrm = state.vrm;
  const rig = state.rig;
  if (!vrm || !rig) return;
  if (!ensureUnitBounds()) return;

  const W = window.innerWidth;
  const H = window.innerHeight;
  // While a window is minimized (or mid-transition) Windows can report a
  // degenerate viewport. Fitting against it would blow up the scale/position,
  // so skip and retry until the real size is back.
  if (!W || !H || W < 8 || H < 8) {
    requestAnimationFrame(fitModel);
    return;
  }

  // The device pixel ratio can change (monitor moves, OS zoom) — keep the
  // drawing buffer in sync or the fit math and the canvas disagree.
  const dpr = window.devicePixelRatio || 1;
  if (renderer.getPixelRatio() !== dpr) renderer.setPixelRatio(dpr);

  renderer.setSize(W, H);
  camera.aspect = W / H;
  camera.updateProjectionMatrix();
  lastFitW = W;
  lastFitH = H;
  lastFitDPR = dpr;

  const b = state.unitBounds;
  const ppu = pxPerUnit();
  const c = containerPx();

  // Scale to fit the window's safe area, then apply the user's scale
  // multiplier.
  const marginX = c.w * FIT_MARGIN_SIDE;
  const marginTop = c.h * FIT_MARGIN_TOP;
  const marginBottom = c.h * FIT_MARGIN_BOTTOM + FIT_BOTTOM_PAD;
  const availW = Math.max(1, c.w - marginX * 2);
  const availH = Math.max(1, c.h - marginTop - marginBottom);
  const scale =
    Math.min(availW / (b.width * ppu), availH / (b.height * ppu)) * state.scaleMul;
  rig.scale.setScalar(scale);

  // Model extent on screen at this scale.
  const widthPx = b.width * scale * ppu;
  const heightPx = b.height * scale * ppu;

  // Feet anchor in window px: window bottom safe margin + user offset,
  // clamped to the window visibility limits (shared with the character drag).
  const anchorX = c.x + c.w / 2;
  const anchorY = c.y + c.h - marginBottom;
  lastFitGeometry = { widthPx, heightPx, anchorX, anchorY };
  const feet = clampFeetPx(
    anchorX + state.charOffset.x * W,
    anchorY + state.charOffset.y * H,
    widthPx,
    heightPx,
    W,
    H,
  );
  const feetPx = feet.x;
  const feetPy = feet.y;

  // Write the clamped anchor back as fractions so stored state matches what
  // is rendered (a maximize can otherwise keep re-amplifying a stale offset).
  const offX = (feetPx - anchorX) / W;
  const offY = (feetPy - anchorY) / H;
  if (
    Math.abs(offX - state.charOffset.x) > 1e-4 ||
    Math.abs(offY - state.charOffset.y) > 1e-4
  ) {
    state.charOffset = { x: offX, y: offY };
    window.overlay.setCharOffset(state.charOffset);
  }

  // Px -> world on the model plane (screen y+ is down, world y+ is up).
  rig.position.set((feetPx - W / 2) / ppu, (H / 2 - feetPy) / ppu, 0);

  // World-space eye position, used by the idle driver to clamp the gaze so
  // the eyeballs never roll to extremes (see IdleDriver.setEyeWorld).
  updateEyeWorld(vrm, rig);
}

const _eyeVec = new THREE.Vector3();

/** Track the eyes' world position (average of both eye bones, when the rig
 * exposes them; otherwise an estimate from the model's height). The screen
 * position of the eyes goes to the idle driver too — her resting gaze spots
 * center on it so the head sits LEVEL instead of dipping toward the window
 * center below her eyes. */
function updateEyeWorld(vrm, rig) {
  if (!state.idle) return;
  const ppu = pxPerUnit();
  const W = window.innerWidth;
  const H = window.innerHeight;
  const nodeL = vrm.humanoid ? vrm.humanoid.getRawBoneNode('leftEye') : null;
  const nodeR = vrm.humanoid ? vrm.humanoid.getRawBoneNode('rightEye') : null;
  if (nodeL && nodeR) {
    nodeL.updateWorldMatrix(true, false);
    nodeL.getWorldPosition(_eyeVec);
    let x = _eyeVec.x;
    let y = _eyeVec.y;
    nodeR.updateWorldMatrix(true, false);
    nodeR.getWorldPosition(_eyeVec);
    x = (x + _eyeVec.x) / 2;
    y = (y + _eyeVec.y) / 2;
    state.idle.setEyeWorld(x, y);
    state.idle.setEyeScreen(W / 2 + x * ppu, H / 2 - y * ppu);
  } else if (state.unitBounds) {
    const x = rig.position.x;
    const y = rig.position.y + state.unitBounds.height * rig.scale.y * 0.92;
    state.idle.setEyeWorld(x, y);
    state.idle.setEyeScreen(W / 2 + x * ppu, H / 2 - y * ppu);
  }
}

/* ------------------------------------------------------------- hit-test --- */

const _hitBox = new THREE.Box3();
const _corner = new THREE.Vector3();

/**
 * Project the model's (rest-pose) world bounds to screen space and cache the
 * result for the overlay's hit-testing. Recomputed once per frame after the
 * render so it follows the body's sway; bone-level poses (arm raises etc.)
 * stay within the rest-pose envelope, which is close enough for clicks.
 */
function updateCharacterRect() {
  const vrm = state.vrm;
  if (!vrm) {
    state.charRect = null;
    return;
  }
  _hitBox.setFromObject(vrm.scene);
  if (_hitBox.isEmpty()) {
    state.charRect = null;
    return;
  }

  const W = window.innerWidth;
  const H = window.innerHeight;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const x of [_hitBox.min.x, _hitBox.max.x]) {
    for (const y of [_hitBox.min.y, _hitBox.max.y]) {
      for (const z of [_hitBox.min.z, _hitBox.max.z]) {
        _corner.set(x, y, z).project(camera);
        const px = ((_corner.x + 1) / 2) * W;
        const py = ((1 - _corner.y) / 2) * H;
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;
      }
    }
  }

  // Tighten the box a bit so clicks near transparent corners pass through.
  const width = maxX - minX;
  const height = maxY - minY;
  const insetX = width * 0.12;
  const insetTop = height * 0.06;
  state.charRect = {
    x: minX + insetX,
    y: minY + insetTop,
    width: width - insetX * 2,
    height: height - insetTop,
  };
}

/* -------------------------------------------------------------- loading --- */
// Models are parsed ONCE and kept resident in memory (modelCache): switching
// characters or restarting never re-downloads/re-parses — the VRM (geometry,
// textures, GPU buffers) is simply re-attached, which is instant. After boot
// the remaining models are preloaded in the background so every switch is a
// zero-latency attach. This is what keeps the companion realtime-responsive.

const MODEL_CACHE_MAX = 4; // resident models before the oldest is evicted
const modelCache = new Map(); // rel -> { vrm, unitBounds }

/** Parse + prepare a VRM once: orientation, bounds, texture sharpness. */
async function buildVrm(url, rel) {
  const loaded = await loader.loadAsync(url);
  const vrm = loaded.userData.vrm;
  if (!vrm) throw new Error(`no VRM data found in ${rel}`);

  // VRM 0.x models face the opposite way — turn her toward the camera.
  VRMUtils.rotateVRM0(vrm);

  // Max anisotropic filtering on every texture: keeps skin/hair/cloth sharp
  // at glancing angles and while she spins — the single biggest quality win.
  const maxAniso = renderer.capabilities.getMaxAnisotropy();
  vrm.scene.traverse((obj) => {
    if (!obj.isMesh) return;
    const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
    for (const mat of materials) {
      if (!mat) continue;
      for (const key of Object.keys(mat)) {
        const value = mat[key];
        if (value && value.isTexture) value.anisotropy = maxAniso;
      }
    }
  });

  // Measure the bounds once, on a clean transform, so fitting is instant and
  // consistent every time this model is attached.
  vrm.scene.position.set(0, 0, 0);
  vrm.scene.updateWorldMatrix(true, true);
  const box = new THREE.Box3().setFromObject(vrm.scene);
  let unitBounds = null;
  if (!box.isEmpty()) {
    unitBounds = {
      minX: box.min.x,
      minY: box.min.y,
      width: box.max.x - box.min.x,
      height: box.max.y - box.min.y,
      cx: (box.min.x + box.max.x) / 2,
    };
  }
  return { vrm, unitBounds };
}

/** Keep the cache bounded; never evict the model currently on stage. */
function cacheModel(rel, entry) {
  modelCache.set(rel, entry);
  if (modelCache.size > MODEL_CACHE_MAX) {
    for (const [key, value] of modelCache) {
      if (modelCache.size <= MODEL_CACHE_MAX) break;
      if (key === state.current) continue;
      modelCache.delete(key);
      VRMUtils.deepDispose(value.vrm.scene);
      console.log(`[vtuber] evicted cached model: ${key}`);
    }
  }
}

// In-flight builds, deduplicated by rel path: a tray switch/double-click can
// request a model that the background preload is already parsing, and two
// parallel buildVrm runs for the same file would leave one full VRM (GPU
// buffers, textures) unreachable and never disposed.
const modelBuilds = new Map(); // rel -> Promise<entry>

function buildVrmOnce(url, rel) {
  let build = modelBuilds.get(rel);
  if (!build) {
    build = buildVrm(url, rel).finally(() => modelBuilds.delete(rel));
    modelBuilds.set(rel, build);
  }
  return build;
}

// Monotonic switch counter: a slow build must not attach its model after a
// NEWER switch has already been requested (last request wins, not last build).
let loadToken = 0;

/** Attach a (cached) VRM to the stage and rebuild the idle driver for it. */
function attachModel(entry, rel) {
  const { vrm } = entry;
  if (state.vrm === vrm) return;

  // Detach the previous model — it stays warm in the cache for instant reuse.
  if (state.vrm) {
    state.idle?.destroy();
    state.idle = null;
    state.body.remove(state.vrm.scene);
  }
  if (!state.rig) {
    // The rig carries the fitted position/scale; the body group inside it is
    // pivoted at her feet and reserved for whole-body idle animations (sways,
    // rises, leans). Built once and reused across model switches.
    state.body = new THREE.Group();
    state.rig = new THREE.Group();
    state.rig.add(state.body);
    scene.add(state.rig);
  }

  state.vrm = vrm;
  state.current = rel;
  state.unitBounds = entry.unitBounds;
  applyRestTransform();
  state.body.add(vrm.scene);
  fitModel();

  state.idle = new IdleDriver(vrm, state.body, {
    gazeToWorld,
    unitSize: {
      width: state.unitBounds ? state.unitBounds.width : 1,
      height: state.unitBounds ? state.unitBounds.height : 1,
    },
  });
  state.idle.setTrackMouse(state.trackMouse);
  fitModel(); // pick up the eye-world position now that the driver exists

  // If the bounds weren't measurable yet, keep retrying the fit over the next
  // frames until the measurement lands, so she never stays mis-positioned.
  // Capped: a model whose bounds never materialize must not spin a
  // world-matrix traversal every frame forever in a 24/7 app.
  if (!state.unitBounds) {
    let attempts = 0;
    const retryFit = () => {
      if (state.vrm !== vrm) return; // a newer model replaced this one
      fitModel();
      if (!state.unitBounds) {
        if (++attempts < 240) {
          requestAnimationFrame(retryFit);
        } else {
          console.warn(`[vtuber] could not measure model bounds: ${rel}`);
        }
      }
    };
    requestAnimationFrame(retryFit);
  }

  const meta = vrm.meta || {};
  console.log(
    `[vtuber] model ready: ${rel}` +
      (meta.name ? ` (${meta.name})` : '') +
      ` ${state.unitBounds ? `${state.unitBounds.width.toFixed(2)}x${state.unitBounds.height.toFixed(2)}` : ''}`,
  );
}

async function loadModel(url, rel, token) {
  let entry = modelCache.get(rel);
  if (entry) {
    console.log(`[vtuber] attaching resident model: ${rel}`);
  } else {
    console.log(`[vtuber] loading model: ${rel}`);
    entry = await buildVrmOnce(url, rel);
    cacheModel(rel, entry);
  }
  if (token !== loadToken) return; // a newer switch superseded this load
  attachModel(entry, rel);
}

/** Warm the cache with every remaining model, one at a time in the
 * background — after this, any tray switch is an instant attach. */
async function preloadAllModels() {
  for (const m of state.models) {
    if (modelCache.has(m.rel)) continue;
    try {
      const entry = await buildVrmOnce(modelUrlFor(m.rel), m.rel);
      if (!modelCache.has(m.rel)) cacheModel(m.rel, entry);
      console.log(`[vtuber] preloaded model: ${m.rel}`);
    } catch (err) {
      console.warn(`[vtuber] preload failed for ${m.rel}:`, err);
    }
  }
}

async function switchModel(rel) {
  if (rel === state.current) return;
  // A tray switch can arrive before boot has populated the model list.
  if (!state.models.length) {
    try {
      const s = await window.overlay.getState();
      state.models = s.models;
    } catch {
      /* fall through — the check below will reject unknown models */
    }
  }
  if (!state.models.some((m) => m.rel === rel)) return;
  const token = ++loadToken;
  try {
    await loadModel(modelUrlFor(rel), rel, token);
  } catch (err) {
    console.error('[vtuber] failed to load model:', err);
  }
}

function cycleModel() {
  if (!state.models.length) return;
  const i = state.models.findIndex((m) => m.rel === state.current);
  const next = state.models[(i + 1) % state.models.length];
  // Go through the main process so tray state + persistence stay in sync;
  // the model is swapped when the change event comes back.
  window.overlay.setModel(next.rel);
}

/* ------------------------------------------------------------- speech --- */
// speak() plays audio through a WebAudio analyser. Lip-sync prefers an
// OFFLINE viseme timeline: the clip is fully known before it plays, so it is
// analyzed once up front (formants classified into the five vowel shapes,
// plosive closures and sibilants detected, everything smoothed with
// lookahead) and the mouth is driven from phoneme identity during playback —
// how far the mouth opens depends on WHICH sound is being said, not on how
// loud it is. The realtime analyser path (loudness-gated formant tracking)
// remains as the fallback for clips that cannot be pre-analyzed.

const speech = {
  context: null,
  audio: null,
  source: null,
  analyser: null,
  data: null,
  freqData: null,
  playing: false,
  clipId: null, // id of the clip currently owned (echoed back on tts-done)
  level: 0,
  jaw: 0, // phoneme-weighted jaw opening (realtime fallback path)
  peak: 0.05, // slow AGC peak so quiet speech still opens the mouth
  lastT: 0,
  f1: 500, // smoothed first formant (Hz) — jaw height of the vowel
  f2: 1500, // smoothed second formant (Hz) — tongue frontness
  // Pre-analyzed viseme timeline for the current clip (offline lip-sync);
  // null when the clip could not be parsed and the realtime path is used.
  timeline: null,
  // Per-frame viseme targets (0..1 each), fed to the idle driver.
  visemes: { aa: 0, ih: 0, ee: 0, oh: 0, ou: 0 },
};

function ensureAudioContext() {
  if (!speech.context) {
    speech.context = new AudioContext();
  }
  return speech.context;
}

/** Play one backend TTS clip with lip-sync. Accepts either a bare url/data
 * URI or the protocol payload object `{ id, wav }` — the id is echoed back
 * with tts-done so the backend can tell a stale confirmation from the clip
 * that is actually playing (echo safety). */
async function speak(payload) {
  const isObj = typeof payload === 'object' && payload !== null;
  const sound = isObj ? payload.wav : payload;
  const clipId = isObj && payload.id != null ? payload.id : null;
  stopSpeaking(); // acks the PREVIOUS clip before the new id takes over
  speech.clipId = clipId;
  if (!sound || !state.idle) {
    // Nothing will ever play this clip — report it done anyway so the voice
    // pipeline never stalls waiting on a confirmation that cannot come.
    stopSpeaking();
    return;
  }
  // Offline lip-sync: the whole clip is known before it plays, so analyze it
  // once up front into a viseme timeline (phoneme identity drives the mouth,
  // not loudness). Falls back to the realtime analyser when the URI is not a
  // parseable WAV (e.g. a streamed url).
  speech.timeline = null;
  try {
    const wav = parseWavDataUri(sound);
    if (wav) speech.timeline = buildVisemeTimeline(wav.samples, wav.sampleRate);
  } catch (err) {
    console.warn('[vtuber] offline lip-sync failed, using realtime:', err);
    speech.timeline = null;
  }
  try {
    const ctx = ensureAudioContext();
    if (ctx.state === 'suspended') await ctx.resume();
    const audio = new Audio(sound);
    audio.crossOrigin = 'anonymous';
    const source = ctx.createMediaElementSource(audio);
    const analyser = ctx.createAnalyser();
    // 1024-point FFT: fine enough bins to resolve formants (~47 Hz/bin at
    // 48 kHz), while the ~21 ms time window still tracks syllables. Low
    // smoothing so the spectrum reacts instead of lagging behind.
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.35;
    source.connect(analyser);
    analyser.connect(ctx.destination);

    speech.audio = audio;
    speech.source = source;
    speech.analyser = analyser;
    speech.data = new Uint8Array(analyser.fftSize);
    speech.freqData = new Uint8Array(analyser.frequencyBinCount);
    speech.playing = true;
    speech.level = 0;
    speech.jaw = 0;
    speech.peak = 0.05;
    speech.f1 = 500;
    speech.f2 = 1500;
    speech.visemes = { aa: 0, ih: 0, ee: 0, oh: 0, ou: 0 };
    speech.lastT = performance.now();
    state.idle.speaking = true;
    // Guard against a stale 'ended' from a previous element: only the element
    // that is currently playing may stop the speech state.
    audio.addEventListener('ended', () => {
      if (speech.audio === audio) stopSpeaking();
    });
    await audio.play();
  } catch (err) {
    console.error('[vtuber] speech playback failed:', err);
    stopSpeaking();
  }
}

function stopSpeaking() {
  const wasActive = speech.playing || Boolean(speech.audio);
  const clipId = speech.clipId;
  if (speech.audio) {
    speech.audio.pause();
    speech.audio.removeAttribute('src');
    speech.audio = null;
  }
  if (speech.source) {
    try {
      speech.source.disconnect();
    } catch {
      /* already disconnected */
    }
    speech.source = null;
  }
  if (speech.analyser) {
    try {
      speech.analyser.disconnect();
    } catch {
      /* already disconnected */
    }
    speech.analyser = null;
  }
  speech.clipId = null;
  speech.timeline = null;
  if (speech.playing || speech.level > 0) {
    speech.playing = false;
    speech.level = 0;
    speech.jaw = 0;
    if (state.idle) {
      state.idle.speaking = false;
      state.idle.setMouthLevel(null);
    }
  }
  // Echo guard: tell the voice backend that her voice is no longer in the
  // air — it keeps the mic gated until this lands, so she never hears
  // herself. Sent for every stop: natural end, replacement, mute, error —
  // and also for clips that were assigned but never played (the backend
  // would otherwise wait out the full grace timeout). The id lets the
  // backend match the confirmation to the clip it is waiting on.
  if (wasActive || clipId != null) window.overlay.ttsDone?.(clipId);
}

/** Mean (0..1) of the analyser's frequency bins inside [f0, f1] Hz. */
function bandEnergy(f0, f1) {
  const nyquist = speech.context.sampleRate / 2;
  const binHz = nyquist / speech.freqData.length;
  const i0 = Math.max(1, Math.floor(f0 / binHz));
  const i1 = Math.min(speech.freqData.length - 1, Math.ceil(f1 / binHz));
  if (i1 < i0) return 0;
  let sum = 0;
  for (let i = i0; i <= i1; i++) sum += speech.freqData[i];
  return sum / ((i1 - i0 + 1) * 255);
}

/**
 * Locate a formant: the strongest peak of the (3-bin smoothed) spectrum
 * inside [f0, f1] Hz, refined to sub-bin accuracy with a parabolic fit.
 * Returns null when the energy is too weak to trust (unvoiced gap).
 */
function findFormantPeak(f0, f1) {
  const nyquist = speech.context.sampleRate / 2;
  const binHz = nyquist / speech.freqData.length;
  const i0 = Math.max(2, Math.floor(f0 / binHz));
  const i1 = Math.min(speech.freqData.length - 3, Math.ceil(f1 / binHz));
  let bestI = -1;
  let bestV = 0;
  for (let i = i0; i <= i1; i++) {
    const v = (speech.freqData[i - 1] + speech.freqData[i] + speech.freqData[i + 1]) / 3;
    if (v > bestV) {
      bestV = v;
      bestI = i;
    }
  }
  if (bestI < 0 || bestV < 24) return null; // ~-20 dBFS floor
  const a = speech.freqData[bestI - 1];
  const b = speech.freqData[bestI];
  const c = speech.freqData[bestI + 1];
  const denom = a - 2 * b + c;
  const off = denom !== 0 ? (0.5 * (a - c)) / denom : 0;
  return (bestI + off) * binHz;
}

/**
 * Per-frame lip-sync. Preferred path: the clip was pre-analyzed into a
 * viseme timeline (see lipsync.mjs) — the mouth is sampled from it at the
 * current playback position, so articulation is phoneme-driven and smoothed
 * with lookahead. Fallback path (no timeline): a realtime analyser tracks
 * the first two formants and classifies the vowel the same way; loudness
 * only opens a saturating gate — it never scales the mouth, so the jaw no
 * longer pumps like a volume bar.
 */
function updateSpeech() {
  if (!speech.playing) return;

  if (speech.timeline && speech.audio) {
    const s = sampleTimeline(speech.timeline, speech.audio.currentTime);
    const vis = speech.visemes;
    for (const name of Object.keys(vis)) vis[name] = s.visemes[name];
    state.idle?.setMouthLevel(s.jaw < 0.02 ? 0 : s.jaw, vis);
    return;
  }

  if (!speech.analyser) return;
  const now = performance.now();
  const dt = Math.min(0.1, Math.max(0.001, (now - speech.lastT) / 1000));
  speech.lastT = now;

  speech.analyser.getByteTimeDomainData(speech.data);
  let sum = 0;
  for (let i = 0; i < speech.data.length; i++) {
    const v = (speech.data[i] - 128) / 128;
    sum += v * v;
  }
  const rms = Math.sqrt(sum / speech.data.length);

  // AGC: the peak decays with a ~2 s half-life, so a passage that gets
  // quieter is re-normalized instead of fading the mouth to a standstill.
  speech.peak = Math.max(rms, speech.peak * Math.pow(0.5, dt / 2), 0.02);
  const target = Math.min(1, Math.pow(rms / (speech.peak * 0.6), 0.8));
  const k = target > speech.level
    ? 1 - Math.exp(-32 * dt) // fast attack
    : 1 - Math.exp(-11 * dt); // gentler release
  speech.level += (target - speech.level) * k;

  const vis = speech.visemes;
  if (speech.level > 0.04) {
    speech.analyser.getByteFrequencyData(speech.freqData);

    // Loudness is a GATE, not a driver: once a frame is comfortably voiced
    // the mouth is fully open, and the vowel identity decides the shape.
    const open = Math.min(1, speech.level / 0.25);

    let targets;
    const eSib = bandEnergy(4000, 8000); // sibilant/fricative hiss
    const eVoice = bandEnergy(70, 1700); // voiced energy
    if (eSib > 0.16 && eSib > eVoice * 0.6) {
      // S/sh/f: teeth nearly together, a small narrow opening.
      const s = Math.min(1, eSib * 3) * 0.32;
      targets = { aa: 0, ih: s * 0.7 * open, ee: s * 0.6 * open, oh: 0, ou: 0 };
    } else {
      // Voiced: track the first two formants (hold the last estimate when a
      // frame has no trustworthy peak, e.g. mid-plosive gap).
      const f1 = findFormantPeak(120, 1100);
      const f2 = findFormantPeak(f1 ? f1 + 250 : 600, 3200);
      if (f1 !== null && f2 !== null) {
        const fk = 1 - Math.exp(-22 * dt); // ~45 ms formant glide
        speech.f1 += (f1 - speech.f1) * fk;
        speech.f2 += (f2 - speech.f2) * fk;
      }
      // Score every vowel by Gaussian distance in (F1, F2) space, sharpened
      // so the closest vowel clearly wins instead of blending to mush.
      let best = 0;
      const scores = {};
      for (const [name, c] of Object.entries(VOWEL_CENTERS)) {
        const d1 = (speech.f1 - c.f1) / 260;
        const d2 = (speech.f2 - c.f2) / 550;
        const sc = Math.exp(-(d1 * d1 + d2 * d2));
        scores[name] = sc;
        if (sc > best) best = sc;
      }
      targets = {};
      const sharpen = 2.5;
      let total = 0;
      for (const name of Object.keys(scores)) {
        const sc = Math.pow(scores[name] / (best || 1), sharpen);
        scores[name] = sc;
        total += sc;
      }
      for (const name of Object.keys(scores)) {
        targets[name] = open * (scores[name] / (total || 1));
      }
    }

    // Glide the viseme mix toward this frame's targets (~60 ms) — real
    // articulation transitions, not per-frame flicker.
    const vk = 1 - Math.exp(-18 * dt);
    let jaw = 0;
    for (const name of Object.keys(vis)) {
      vis[name] += ((targets[name] || 0) - vis[name]) * vk;
      if (vis[name] < 0.004) vis[name] = 0;
      jaw += vis[name] * (VOWEL_OPEN[name] || 0);
    }
    // Jaw follows the weighted vowel openness, not the raw loudness.
    speech.jaw = Math.min(1, jaw * 1.15);
  } else {
    // Silence between words: everything eases to closed (the idle driver's
    // release damping finishes the job).
    for (const name of Object.keys(vis)) vis[name] = 0;
    speech.jaw = 0;
  }

  state.idle?.setMouthLevel(speech.jaw < 0.02 ? 0 : speech.jaw, vis);
}

/* ---------------------------------------------------------- render loop --- */

const timer = new THREE.Timer();
// Page Visibility API support: no giant deltas after the window was hidden.
timer.connect(document);
renderer.setAnimationLoop((timestamp) => {
  timer.update(timestamp);
  // Clamp delta so a paused/background window doesn't teleport the animations.
  const delta = Math.min(timer.getDelta(), 0.05);

  // Belt-and-braces resize handling: if the viewport (or DPI) differs from
  // what the fit assumes — which can happen when a minimize/maximize cycle
  // skips or reorders resize events — refit right now. This is what keeps the
  // character centered and fully visible after the window is restored.
  const W = window.innerWidth;
  const H = window.innerHeight;
  const dpr = window.devicePixelRatio || 1;
  if (state.vrm && W >= 8 && H >= 8 && (W !== lastFitW || H !== lastFitH || dpr !== lastFitDPR)) {
    fitModel();
    editUI.refresh();
  }

  if (state.vrm) {
    updateSpeech();
    // Idle first: it writes this frame's expressions, bone rotations, gaze
    // target and body pose. vrm.update() then copies the normalized bones to
    // the rig, compensates the eyes for the head turn, and applies the
    // expressions and spring bones.
    state.idle?.applyFrame(delta);
    state.vrm.update(delta);
  }
  renderer.render(scene, camera);
  updateCharacterRect();
});

// Refit whenever the window/canvas changes size. Fit immediately, then again
// on the next frame so any layout settle is caught — the character must
// never drift or get the face cut off on maximize/restore. Focus and
// visibility changes cover restore-from-minimized cases where a resize event
// arrives late or not at all.
const refit = () => {
  fitModel();
  editUI.refresh();
  requestAnimationFrame(fitModel);
};
window.addEventListener('resize', refit);
window.addEventListener('focus', refit);
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) refit();
});

/* ------------------------------------------------------- overlay wiring --- */

// The main process streams the global cursor position (window-relative), so
// the eyes keep following the pointer even when it leaves the overlay window.
// The driver ignores the stream while mouse tracking is turned off.
window.overlay.onCursor(({ x, y }) => state.idle?.setPointer(x, y));

// Container/character edit modes. The window itself is the container:
// 'container' mode resizes/moves the actual window (bounds go to the main
// process in screen coords), 'character' mode moves her inside it. The
// character offset is stored as fractions of the window so it survives
// window resizes.
const editUI = setupEditMode({
  getRect: containerPx,
  getWindowBounds: () => ({
    x: window.screenX,
    y: window.screenY,
    w: window.innerWidth,
    h: window.innerHeight,
  }),
  onWindowBoundsChanged: (bounds) => window.overlay.setWindowBounds(bounds),
  getCharOffset: () => ({
    x: state.charOffset.x * window.innerWidth,
    y: state.charOffset.y * window.innerHeight,
  }),
  // Same visibility limits the rendering uses — keeps the drag symmetric
  // (the character never lags behind or detaches from the cursor).
  clampCharOffset: clampCharOffsetPx,
  onCharOffsetChanged: (off) => {
    state.charOffset = {
      x: off.x / window.innerWidth,
      y: off.y / window.innerHeight,
    };
    fitModel();
    window.overlay.setCharOffset(state.charOffset);
  },
});

function setEditMode(mode) {
  state.editMode = mode;
  editUI.setMode(mode);
}

// The tray can flip mouse tracking and edit modes too — stay in sync.
window.overlay.onStateChanged(({ trackMouse, editMode, voice }) => {
  if (typeof trackMouse === 'boolean') {
    state.trackMouse = trackMouse;
    state.idle?.setTrackMouse(trackMouse);
  }
  if (typeof editMode === 'string') setEditMode(editMode);
  // Voice switched off while she is talking — stop the current clip.
  if (voice === false) stopSpeaking();
});

setupOverlay({
  getCharacterRect: () => state.charRect,
  overEditUI: (x, y) => editUI.isOver(x, y),
  onCycleModel: cycleModel,
  onPoke: () => state.idle?.poke(),
  onScale: (factor) => {
    state.scaleMul = Math.min(2.5, Math.max(0.4, state.scaleMul * factor));
    fitModel();
  },
});

/**
 * Bridge for the assistant.py backend: the main process spawns the voice
 * pipeline and forwards its `backend:event` IPC payloads here.
 */
window.companion = {
  /** 'idle' | 'listening' | 'thinking' | 'speaking' — colors her posture
   * and gaze through the idle driver (attentive while listening, look-up
   * while thinking; speaking is owned by lip-sync + talking gestures). */
  setState(value) {
    const known = ['idle', 'listening', 'thinking', 'speaking'];
    if (known.includes(value)) {
      state.idle?.setBackendState(value);
    }
  },
  /** Play audio with automatic lip-sync. Accepts a bare http(s) url / data:
   * URI or the backend payload object `{ id, wav }` (the id is echoed back
   * with tts-done for the backend's echo-safety handshake). */
  speak(sound) {
    return speak(sound);
  },
  stopSpeaking() {
    stopSpeaking();
  },
  /** Conversation emotion from the backend ('happy' | 'smug' | 'evil' |
   * 'angry' | 'sad' | 'surprised' | ...) — she acts it with her face. */
  setEmotion(value) {
    state.idle?.setEmotion(value);
  },
};

window.overlay.onBackendEvent((event) => {
  if (!event) return;
  if (event.type === 'state') window.companion.setState(event.value);
  else if (event.type === 'speak') window.companion.speak(event.value);
  else if (event.type === 'emotion') window.companion.setEmotion(event.value);
});

window.overlay.onModelChanged(({ rel }) => switchModel(rel));

/* ------------------------------------------------------------------ boot --- */

(async () => {
  try {
    const { models, model, trackMouse, charOffset } = await window.overlay.getState();
    state.models = models;
    state.trackMouse = trackMouse !== false;
    if (charOffset && Number.isFinite(charOffset.x) && Number.isFinite(charOffset.y)) {
      state.charOffset = charOffset;
    }
    console.log(`[vtuber] models found: ${models.map((m) => m.name).join(' | ') || '(none)'}`);
    if (model) await switchModel(model);
    else console.warn('[vtuber] no models found under characters_models/');
    // Warm the rest of the cache in the background — later switches attach
    // instantly instead of parsing mid-interaction.
    preloadAllModels();
  } catch (err) {
    console.error('[vtuber] boot failed:', err);
  } finally {
    // Tell the main process the renderer is ready to receive backend
    // events — the voice pipeline is started only after this lands.
    window.overlay.ready?.();
  }
})();
