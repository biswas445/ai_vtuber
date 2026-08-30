/**
 * Idle behavior driver for VRM companions (three-vrm): everything that keeps
 * her alive between interactions. VRM models ship with no idle motions of
 * their own (and this one is stored in a T-pose), so the driver first applies a
 * calm resting pose — arms lowered to her sides, elbows softly bent — and
 * then keeps her alive with subtle, continuous motion. Random triggered
 * animations (head tilts, expression pops, arm movements, marches, spins...)
 * were removed by request; what remains is:
 *
 * - Always-on signs of life: a soft breathing bob with a hint of chest rise,
 *   a slow micro sway that also travels through the hanging arms, and two
 *   slow incommensurate head micro-drifts, so she is never perfectly still.
 *   Blinks always close BOTH eyes together from a single shared value — the
 *   driver never closes one eye on its own.
 * - Reactions to being poked (clicked on): a quick giggle, surprised
 *   double-take, curious head tilt or a little hop.
 *
 * Gaze: with mouse tracking enabled she follows the global cursor via the
 * VRM lookAt target (and even steals the occasional glance away so she
 * doesn't stare); with it disabled the cursor is ignored entirely and she
 * wanders around on her own using saccade-like hops between resting spots.
 * The head follows the gaze with a damped lag — eyes lead, head follows —
 * split across head/neck/spine so turns feel articulated, not robotic. At
 * rest the head sits LEVEL: resting gaze spots center on her own eye line,
 * so she looks straight at the viewer instead of dipping her chin.
 *
 * While the backend says she is speaking she layers lecturer-style talking
 * gestures — arms working in front of her with bent wrists and open palms,
 * head nods, a light sway — held up for the whole conversation so they do
 * not slump on every gap between sentence clips. The mouth itself follows
 * the audio via phoneme-driven lip-sync from the renderer. The backend can
 * also push the emotion of the current reply (happy/smug/evil/angry/
 * sad/surprised), which she acts with her face at a softened weight that
 * settles to a sustain, so the eyes never squint-shut for a whole reply.
 *
 * Bone rotations are written onto the NORMALIZED humanoid bones every frame
 * (clean base pose first, then layers on top), using the same face-front
 * conjugation VRM's own lookAt bone applier uses, so the signs stay correct
 * for both VRM 0.x and 1.0 models regardless of their internal bone
 * orientation. VRMLookAt keeps compensating the eye bones while the head
 * rotates, so the gaze stays locked on target.
 *
 * Mouth: the backend bridge can override the mouth opening directly, and
 * speak() audio drives it via lip-sync from the renderer; otherwise the
 * mouth stays gently closed.
 */

import * as THREE from 'three';

const BLINK_CLOSE_MS = 90;
const BLINK_HOLD_MS = 50;
const BLINK_OPEN_MS = 190;
// How long the model keeps looking at the cursor's last position after it
// stops moving, before she starts glancing around on her own.
const WANDER_AFTER_MS = 3500;
// When a poke interrupts an action, the interrupted action keeps
// contributing while it decays over this window (crossfaded with the
// reaction), so nothing ever snaps in a single frame.
const POKE_FADE_MS = 280;
// Same idea for conversation emotions: a new emotion ramps in while the
// outgoing one fades out over this window, so the face never snaps.
const EMOTION_CROSSFADE_MS = 380;

// Head-follow behavior: the gaze direction (already clamped to a natural
// eye range around the eyeballs, see clampGaze) turns the head chain with a
// damped lag — the VRM lookAt keeps the eyes on target, so this fraction is
// what makes her turn toward you instead of staring sideways with eyeballs
// alone. The head stays LEVEL: it may dip a little toward whatever she
// looks at, but it never cranes up and never bows down.
const HEAD_YAW_MAX = 0.42; // rad (~24 deg) — beyond this the eyes take over
// Pitch range is deliberately asymmetric and small: a few degrees up at most
// (no upward lock), a small comfortable dip down.
const HEAD_PITCH_UP_MAX = 0.07; // rad (~4 deg) above level
const HEAD_PITCH_DOWN_MAX = 0.13; // rad (~7 deg) below level
const HEAD_FOLLOW_FRACTION = 0.45; // how much of the gaze pitch/yaw the head joins
const HEAD_FOLLOW_RATE = 2.4; // exponential damping rate (1/s) — calm turns
// Neutral head pitch: exactly LEVEL. Resting gaze spots sit on her eye
// line, so at rest she looks straight at the viewer — chin neither dipped
// nor lifted.
const NEUTRAL_HEAD_PITCH = 0.0;
// Hard limits for the composed head rotation written to the bones — whatever
// actions layer on top can never lock her looking upward.
const HEAD_PITCH_HARD_MIN = -0.1; // ~6 deg up, transient actions only
const HEAD_PITCH_HARD_MAX = 0.32;
const HEAD_YAW_HARD_MAX = 0.6;
// Eye range: how far the eyeballs may roll inside the head (rad). Looking up
// is limited so the irises never roll under the upper lids (which reads as a
// stuck upward stare / half-closed eye); down and sideways stay comfortable.
const EYE_PITCH_UP_MAX = 0.3; // ~17 deg up
const EYE_PITCH_DOWN_MAX = 0.52; // ~30 deg down
const EYE_YAW_MAX = 0.6; // ~34 deg sideways
// Split of the head-chain rotation across bones so turns feel articulated.
const HEAD_SPLIT = { head: 0.6, neck: 0.25, spine: 0.15 };

// Resting pose (radians, applied on top of the stored T-pose). Arms hang to
// her sides with a little daylight under the armpits, elbows softly bent
// forward, so she stands naturally instead of striking a T-pose.
const REST_ARM_ROLL = 1.35; // left arm: -roll, right arm: +roll (raises are +/-)
const REST_ELBOW_YAW = 0.12; // left elbow: -yaw, right elbow: +yaw (bends forward)
// Hands are never held rigid: wrists keep a soft inward bend and the fingers
// a light relaxed curl at all times, stronger during effort (gestures).
// Kept subtle so the exact axis reads as a natural cup, not a bend.
const REST_WRIST_PITCH = 0.12; // wrist flex, toward the palm side
const REST_FINGER_CURL = 0.2; // gentle curl shared by all finger joints

/** Conversation emotions (pushed by the backend) mapped onto VRM expression
 * weights. She holds the face for the duration of the utterance, then it
 * fades out (see applyFrame). Weights are kept LOW on purpose: this model's
 * "happy" squints the eyes and "angry" pulls the brows, and a full-strength
 * expression held for a whole utterance reads as a frozen grimace while she
 * talks. The face also settles to a softer sustain mid-utterance (see
 * applyFrame) instead of holding the peak. Only expressions the model
 * actually has are applied. */
const EMOTION_EXPRESSIONS = {
  happy: { happy: 0.5 },
  smug: { happy: 0.38, relaxed: 0.3 },
  evil: { angry: 0.22, happy: 0.28 },
  angry: { angry: 0.5 },
  sad: { sad: 0.55 },
  surprised: { surprised: 0.6 },
};
// The emotion shows at full weight when it first lands, then settles to this
// fraction over EMOTION_SETTLE_MS — a natural face relaxes while talking.
const EMOTION_SUSTAIN = 0.65;
const EMOTION_SETTLE_MS = 1600;

function smoothstep(a, b, t) {
  const x = Math.min(1, Math.max(0, (t - a) / (b - a)));
  return x * x * (3 - 2 * x);
}

const clamp01 = (x) => Math.min(1, Math.max(0, x));
const easeInQuad = (t) => t * t;

/** Keep a resting gaze spot (window px y) inside a comfortable vertical
 * range even when the eyes sit near a window edge. */
function clampGazeSpotY(y, h) {
  return Math.min(h * 0.9, Math.max(h * 0.08, y));
}

/** Blend an interrupted action's bucket into the current frame's bucket:
 * every channel the old action wrote becomes old*wOld + current*(1-wOld) —
 * a true crossfade, so the hand-off can never snap. Channels only the new
 * action writes are untouched; channels only the old one wrote decay to 0. */
function mergeBucket(dst, src, wOld) {
  const wNew = 1 - wOld;
  for (const key of Object.keys(src)) {
    const value = src[key];
    if (typeof value !== 'number') continue;
    dst[key] = value * wOld + (dst[key] || 0) * wNew;
  }
}

// Mirrors VRMLookAtBoneApplier._getWorldFaceFrontQuat: the quaternion that
// carries +Z onto the model's face-front direction. Conjugating a yaw/pitch
// euler by it produces the correct normalized-bone rotation for both VRM 0.x
// (face-front -Z) and VRM 1.0 (face-front +Z) models.
function faceFrontQuaternion(faceFront, target) {
  if (faceFront.distanceToSquared(new THREE.Vector3(0, 0, 1)) < 0.01) {
    return target.identity();
  }
  const azimuth = Math.atan2(-faceFront.z, faceFront.x);
  const altitude = Math.atan2(faceFront.y, Math.hypot(faceFront.x, faceFront.z));
  const euler = new THREE.Euler(0, Math.PI / 2 + azimuth, altitude, 'YZX');
  return target.setFromEuler(euler);
}

// Scratch objects — reused every frame instead of allocating.
const _euler = new THREE.Euler(0, 0, 0, 'YXZ');
const _quat = new THREE.Quaternion();
const _gazeVec = new THREE.Vector3();

export class IdleDriver {
  /**
   * @param {import('@pixiv/three-vrm').VRM} vrm The loaded VRM model.
   * @param {THREE.Group} body The group carrying the model, pivoted at her
   * feet — whole-body poses animate its transform.
   * @param {{ gazeToWorld: (xPx: number, yPx: number, out: THREE.Vector3) => THREE.Vector3, unitSize: { width: number, height: number } }} fit
   * Fit info from the renderer: maps window px to world coords on the model
   * plane, plus the model's unit-scale size so amplitudes scale with her.
   */
  constructor(vrm, body, fit) {
    this.vrm = vrm;
    this.body = body;
    this.fit = fit;

    this.expressions = vrm.expressionManager || null;

    // Many VRM models ship SINGLE-EYE blink expressions (blink_l / blink_r).
    // Nothing in this driver may ever close one eye on its own — a wink reads
    // as a broken blink — so both lids are always driven from the one shared
    // blink value: applyBlink pins the per-eye channels to zero whenever a
    // unified 'blink' exists, and drives both with the same weight otherwise.
    // The per-eye expressions must stay REGISTERED: unregistering them would
    // make setValue a silent no-op, leaving models that lack a unified
    // 'blink' with no blink at all (a permanent unblinking stare).

    // Normalized humanoid bones we animate, when the model exposes them.
    this.bones = {};
    for (const name of [
      'head',
      'neck',
      'spine',
      'chest',
      'upperChest',
      'hips',
      'leftShoulder',
      'rightShoulder',
      'leftUpperArm',
      'rightUpperArm',
      'leftLowerArm',
      'rightLowerArm',
      'leftHand',
      'rightHand',
      'leftUpperLeg',
      'rightUpperLeg',
      'leftLowerLeg',
      'rightLowerLeg',
      'leftFoot',
      'rightFoot',
    ]) {
      const node = vrm.humanoid ? vrm.humanoid.getNormalizedBoneNode(name) : null;
      if (node) this.bones[name] = node;
    }

    // Finger chains (proximal → intermediate → distal) collected per side so
    // the hands can keep a soft relaxed curl instead of rigid flat paddles.
    this.fingers = { left: [], right: [] };
    if (vrm.humanoid) {
      for (const side of ['left', 'right']) {
        for (const finger of ['Thumb', 'Index', 'Middle', 'Ring', 'Little']) {
          const chain = [];
          for (const joint of ['Proximal', 'Intermediate', 'Distal']) {
            const node = vrm.humanoid.getNormalizedBoneNode(`${side}${finger}${joint}`);
            if (node) chain.push(node);
          }
          if (chain.length) this.fingers[side].push(chain);
        }
      }
    }

    // The VRM lookAt steers the eye bones (or lookAt expressions) toward this
    // target object; we move the target instead of the eyes. updateGaze
    // re-aims it every frame before the first render; the fallback spot is
    // the world-space screen center (around/below her eye level once fitted),
    // so even a stray early frame looks gently at the viewer, never up.
    this.lookAtTarget = new THREE.Object3D();
    this.lookAtTarget.position.set(0, 0, 0.5);
    if (vrm.lookAt) {
      vrm.lookAt.autoUpdate = true;
      vrm.lookAt.target = this.lookAtTarget;
    }

    // Face-front conjugation for bone rotations (see faceFrontQuaternion).
    const faceFront = vrm.lookAt ? vrm.lookAt.faceFront : new THREE.Vector3(0, 0, 1);
    this.faceFrontQuat = faceFrontQuaternion(faceFront, new THREE.Quaternion());
    this.faceFrontQuatInv = this.faceFrontQuat.clone().invert();

    // Eyes' world position, fed by the renderer after each fit (see
    // setEyeWorld). The gaze target is clamped around this point so the
    // eyeballs never roll to extremes. Defaults to a sensible spot until the
    // first fit arrives. eyePx is the same point in WINDOW px — resting gaze
    // spots (wander, glances) center on it so the head rests level.
    this.eyeWorld = new THREE.Vector3(0, 0.5, 0);
    this.eyePx = { x: window.innerWidth / 2, y: window.innerHeight * 0.42 };
    /** Clamped gaze direction of the current frame (rad; yaw+ = her left,
     * pitch+ = up), computed while aiming the lookAt target. */
    this.lookYaw = 0;
    this.lookPitch = 0;

    // Custom expressions are discovered lazily by name (models name them
    // freely — "Surprised", "びっくり", ...).
    this._customCache = {};

    /** The facial action currently playing, if any. Random scheduled
     * actions were removed by request — only poke reactions use this. */
    this.action = null;

    /** Whole-body pose channel — likewise only fed by poke reactions. */
    this.bodyAction = null;

    // Poke snap guard: when a poke replaces an action mid-flight, the
    // interrupted action is kept here and its contributions decay over
    // POKE_FADE_MS while the reaction crossfades in (see applyFrame).
    this.fadingFace = null; // { action, t0 }
    this.fadingBody = null; // { action, t0 }

    this.blink = {
      phase: 'open',
      nextAt: performance.now() + 1200 + Math.random() * 2000,
      t0: 0,
      /** Normalized 0..1 lid openness computed for the current frame. */
      open: 1,
    };

    /** Lip-sync level (0..1) fed by the renderer while speak() audio plays. */
    this.mouthLevel = null;
    /** Per-vowel mouth-shape targets (0..1 each) from the renderer's
     * formant-based lip-sync; the viseme mixer glides toward these. */
    this.visemeTargets = null;
    /** True while speak() audio is playing — suppresses mouth-owned actions. */
    this.speaking = false;
    /**
     * Viseme mixer: current weights of each mouth shape, damped toward their
     * targets every frame so the mouth glides between shapes instead of
     * flapping. 'aa' = open, 'ih'/'ee' = wide, 'oh'/'ou' = rounded.
     */
    this.visemes = { aa: 0, ih: 0, ee: 0, oh: 0, ou: 0 };

    /**
     * Talking-gesture channel: eases 0→1 while speaking and back out when
     * speech ends; all speech gestures are multiplied by it so they blend in
     * and out gently. The phase accumulates slowly and drives varied,
     * non-repeating hand/head motion.
     */
    this.talkBlend = 0;
    this.talk = {
      phase: Math.random() * 100,
      // Per-gesture flavor picked fresh for each utterance so no two talking
      // stretches move quite alike (re-rolled when speech starts, see
      // updateTalking).
      seedA: Math.random() * Math.PI * 2,
      seedB: Math.random() * Math.PI * 2,
      amp: 0.8 + Math.random() * 0.4,
    };
    /** Previous frame's talking flag — detects conversation starts. */
    this._wasTalking = false;

    /**
     * Conversation emotion pushed by the backend (see setEmotion): the
     * expression weights she acts with, held while the utterance plays and
     * fading out after `emotionUntil`.
     */
    this.emotionWeights = null;
    this.emotionUntil = 0;
    this.emotionSetAt = 0;
    /** Emotion replaced mid-flight by a newer one — faded out over
     * EMOTION_CROSSFADE_MS so the face never snaps between emotions. */
    this.prevEmotion = null; // { weights, t0 }

    /** Whether the gaze follows the cursor (see setTrackMouse). */
    this.trackMouse = true;

    this.pointer = { x: window.innerWidth / 2, y: window.innerHeight * 0.5, at: 0 };
    // Resting gaze spot used while wandering; retargeted every so often so
    // the eyes make natural saccade hops instead of drifting continuously.
    this.wander = { x: window.innerWidth / 2, y: window.innerHeight * 0.5, nextAt: 0 };
    // Brief looks away from the cursor while tracking, so she doesn't stare.
    this.glance = {
      x: 0,
      y: 0,
      until: 0,
      nextAt: performance.now() + 8000 + Math.random() * 8000,
    };
    // Last computed gaze target (window-relative px); actions may offset it.
    this.gaze = { x: window.innerWidth / 2, y: window.innerHeight * 0.5 };
    /** Smoothed world-space gaze point the lookAt target chases. Real
     * saccades are fast but never instantaneous — this glides the target
     * over ~40 ms so wander hops and glances read as real eye movements
     * instead of teleports (see applyFrame). */
    this.gazePos = null; // Vector3, created on first use

    // Assistant backend state ('idle' | 'listening' | 'thinking' |
    // 'speaking'), fed by the renderer bridge. The blends ease 0..1 toward
    // the active state every frame so posture/gaze coloring never pops.
    this.backendState = 'idle';
    this.stateBlend = { listening: 0, thinking: 0 };
    // Slowly drifting upper spot the gaze leans on while thinking — the
    // classic "hmm, let me see" look-up.
    this.think = {
      x: window.innerWidth * 0.3,
      y: window.innerHeight * 0.2,
      nextAt: 0,
    };

    // Damped head-follow state (radians, VRM convention: yaw+ = her left,
    // pitch+ = down). Pitch starts level.
    this.headYaw = 0;
    this.headPitch = NEUTRAL_HEAD_PITCH;

    this._onMouseMove = (e) => {
      if (!this.trackMouse) return;
      this.pointer.x = e.clientX;
      this.pointer.y = e.clientY;
      this.pointer.at = performance.now();
    };
    window.addEventListener('mousemove', this._onMouseMove);

    // Hair/cloth physics: VRM spring bones are simulated by three-vrm inside
    // vrm.update() — they react to body and head movement with delay, sway and
    // damping on their own, so secondary motion comes for free when they
    // exist. Log the count so a model without springs is easy to spot.
    const springCount =
      vrm.springBoneManager && vrm.springBoneManager.springBones
        ? vrm.springBoneManager.springBones.length
        : 0;

    console.log(
      `[vtuber] idle driver ready (bones: ${Object.keys(this.bones).join(', ') || 'none'}; ` +
        `expressions: ${
          this.expressions
            ? this.expressions.expressions.map((e) => e.expressionName).join(', ')
            : 'none'
        }; spring bones: ${springCount})`,
    );
  }

  /* ------------------------------------------------------- expressions --- */

  hasExpression(name) {
    return Boolean(this.expressions && this.expressions.getExpression(name));
  }

  setExpression(name, weight) {
    if (!this.expressions) return;
    this.expressions.setValue(name, weight);
  }

  /** Find an expression whose name matches, custom ones included (cached). */
  findExpression(re) {
    const key = re.source;
    if (key in this._customCache) return this._customCache[key];
    let found = null;
    if (this.expressions) {
      for (const expression of this.expressions.expressions) {
        if (re.test(expression.expressionName)) {
          found = expression.expressionName;
          break;
        }
      }
    }
    this._customCache[key] = found;
    return found;
  }

  /* -------------------------------------------------------------- blink --- */

  updateBlink(now) {
    const b = this.blink;
    // `open` is a normalized 0..1 lid openness for THIS frame. The state
    // machine always runs a blink to completion (closing → hold → opening) —
    // nothing resets it mid-way, so a blink can never be cut short.
    switch (b.phase) {
      case 'open':
        b.open = 1;
        if (now >= b.nextAt) {
          b.phase = 'closing';
          b.t0 = now;
        }
        break;
      case 'closing': {
        const p = clamp01((now - b.t0) / BLINK_CLOSE_MS);
        b.open = 1 - easeInQuad(p);
        if (p >= 1) {
          b.phase = 'hold';
          b.t0 = now;
        }
        break;
      }
      case 'hold':
        b.open = 0;
        if (now - b.t0 >= BLINK_HOLD_MS) {
          b.phase = 'opening';
          b.t0 = now;
        }
        break;
      case 'opening': {
        const p = clamp01((now - b.t0) / BLINK_OPEN_MS);
        b.open = smoothstep(0, 1, p);
        if (p >= 1) {
          b.phase = 'open';
          // Natural, slightly irregular blink cadence — frequent enough that
          // she never looks like she stopped blinking. People blink noticeably
          // MORE while talking (roughly every 1–2 s), so the speaking gaps are
          // tighter; roughly 1 in 8 blinks is a quick double.
          const base = this.speaking ? 850 : 1800;
          const spread = this.speaking ? 1300 : 3200;
          const gap = base + Math.random() * spread;
          b.nextAt = now + (Math.random() < 0.12 ? 220 + Math.random() * 260 : gap);
        }
        break;
      }
    }

    b.closed = 1 - b.open;
    return b.closed;
  }

  /**
   * Write the blink as the FINAL, authoritative eye value. Both eyes are
   * always driven from the one shared `closed` number, so they can never
   * desync, and because this runs after every other expression nothing can
   * override or reopen the eyes mid-blink. `extraClosed` folds in squints.
   */
  applyBlink(extraClosed = 0) {
    const closed = Math.min(1, (this.blink.closed || 0) + extraClosed);
    if (this.hasExpression('blink')) {
      this.setExpression('blink', closed);
      // Pin any per-eye channels to zero so stale or outside writes can never
      // close a single eye on their own.
      if (this.hasExpression('blinkLeft')) this.setExpression('blinkLeft', 0);
      if (this.hasExpression('blinkRight')) this.setExpression('blinkRight', 0);
    } else {
      // Rig only carries per-eye blinks: drive them with the SAME value.
      this.setExpression('blinkLeft', closed);
      this.setExpression('blinkRight', closed);
    }
  }

  /* -------------------------------------------------------------- mouth --- */

  /**
   * Multi-viseme lip sync. The renderer tracks the voice's first two
   * formants and sends per-vowel mouth-shape targets every frame; the mixer
   * here glides each viseme toward its target with a fast attack and a
   * gentler release, so the mouth shapes each sound like real articulation
   * instead of flapping. With no speech at all, everything eases closed.
   */
  updateMouth(delta) {
    const v = this.visemes;
    let targets;
    if (this.mouthLevel != null) {
      // Lip-sync owns the mouth while audio plays.
      targets = this.visemeTargets || {
        aa: clamp01(this.mouthLevel),
        ih: 0,
        ee: 0,
        oh: 0,
        ou: 0,
      };
    } else {
      targets = null; // no speech — everything glides to closed
    }

    // Damp each viseme toward its target (fast attack, gentle release).
    const attack = 1 - Math.exp(-26 * delta);
    const release = 1 - Math.exp(-11 * delta);
    for (const name of Object.keys(v)) {
      const target = targets ? targets[name] || 0 : 0;
      const k = target > v[name] ? attack : release;
      v[name] += (target - v[name]) * k;
      if (v[name] < 0.004) v[name] = 0;
      if (this.hasExpression(name)) this.setExpression(name, v[name]);
    }
  }

  /* --------------------------------------------------------------- gaze --- */

  /** Enables/disables cursor tracking. While disabled the pointer is neither
   * captured nor followed — she just wanders around on her own. */
  setTrackMouse(enabled) {
    if (this.trackMouse === enabled) return;
    this.trackMouse = enabled;
    if (!enabled) {
      this.pointer.at = 0; // invalidate the last cursor position right away
      this.glance.until = 0;
      this.wander.nextAt = 0; // pick a fresh wander spot on the next frame
    } else {
      this.glance.nextAt = performance.now() + 6000 + Math.random() * 8000;
    }
  }

  /** Feed a gaze target (window-relative coords) from outside, e.g. the
   * global cursor stream — lets the eyes follow the pointer off-window. */
  setPointer(x, y) {
    if (!this.trackMouse) return;
    this.pointer.x = x;
    this.pointer.y = y;
    this.pointer.at = performance.now();
  }

  /** Assistant state from the backend bridge ('idle' | 'listening' |
   * 'thinking' | 'speaking') — colors her posture and gaze (see
   * updateStateBlend). 'speaking' needs no coloring here: lip-sync and the
   * talking gestures already own it. */
  setBackendState(value) {
    if (this.backendState === value) return;
    this.backendState = value;
  }

  /** Ease the listening/thinking blends toward the active backend state. */
  updateStateBlend(delta) {
    const k = 1 - Math.exp(-3 * delta);
    const b = this.stateBlend;
    b.listening += ((this.backendState === 'listening' ? 1 : 0) - b.listening) * k;
    b.thinking += ((this.backendState === 'thinking' ? 1 : 0) - b.thinking) * k;
  }

  /** Lip-sync from the renderer: `value` = jaw level (0..1), `visemes` =
   * per-vowel mouth-shape targets. With the offline timeline path these are
   * phoneme-driven (vowel identity sets the opening, not loudness); pass
   * null to release the mouth. */
  setMouthLevel(value, visemes) {
    this.mouthLevel = value;
    if (visemes) this.visemeTargets = visemes;
    else if (value == null) this.visemeTargets = null;
  }

  /** Eyes' world position from the renderer (recomputed on every fit). */
  setEyeWorld(x, y) {
    this.eyeWorld.set(x, y, this.eyeWorld.z);
  }

  /** Eyes' SCREEN position (window px) — resting gaze spots center on this
   * line so the head rests level instead of dipping toward the window
   * center. */
  setEyeScreen(px, py) {
    this.eyePx.x = px;
    this.eyePx.y = py;
  }

  /** Conversation emotion from the backend ('happy' | 'smug' | 'evil' |
   * 'angry' | 'sad' | 'surprised' | 'neutral'). She acts it with her face
   * for the duration of the utterance (kept alive while speaking, see
   * applyFrame), then the face fades back to neutral. */
  setEmotion(tag) {
    const weights = EMOTION_EXPRESSIONS[tag] || null;
    const now = performance.now();
    // A new emotion arriving while the old one is still showing (or still
    // fading) becomes a short crossfade instead of a one-frame swap.
    if (this.emotionWeights && now < this.emotionUntil + 1200) {
      this.prevEmotion = { weights: this.emotionWeights, t0: now };
    }
    this.emotionWeights = weights;
    this.emotionSetAt = now;
    this.emotionUntil = now + (weights ? 6000 : 0);
  }

  /** Pick this frame's gaze target (window px). Head-follow happens later,
   * after the target has been mapped to world space and clamped to a natural
   * eye range (see clampGaze + updateHeadFollow). */
  updateGaze(now) {
    const w = window.innerWidth;
    const h = window.innerHeight;
    let tx;
    let ty;

    const listening = this.stateBlend.listening > 0.35;
    const pointerFresh = this.pointer.at > 0 && now - this.pointer.at < WANDER_AFTER_MS;
    if (this.trackMouse && (pointerFresh || listening)) {
      // Steal a brief glance away from the cursor every now and then — but
      // never while listening to the user: then she holds eye contact.
      if (!listening && now >= this.glance.nextAt) {
        this.glance.x = w * (0.1 + Math.random() * 0.8);
        // Glance spots sit in a comfortable band around her OWN eye line, so
        // even a glance away keeps the head level.
        this.glance.y = clampGazeSpotY(this.eyePx.y + h * (-0.08 + Math.random() * 0.16), h);
        this.glance.until = now + 500 + Math.random() * 900;
        this.glance.nextAt = now + 9000 + Math.random() * 10000;
      }
      if (!listening && now < this.glance.until) {
        tx = this.glance.x;
        ty = this.glance.y;
      } else {
        tx = this.pointer.x;
        ty = this.pointer.y;
      }
    } else {
      // Wander: hop between random resting spots (saccade-style) instead of
      // drifting on a fixed path — reads much more like real eyes. Spots sit
      // in a band around her OWN eye line so the resting gaze (and the head
      // that follows it) stays level — neither staring at the floor nor
      // craning upward.
      if (now >= this.wander.nextAt) {
        this.wander.x = w * (0.2 + Math.random() * 0.6);
        this.wander.y = clampGazeSpotY(this.eyePx.y + h * (-0.06 + Math.random() * 0.12), h);
        this.wander.nextAt = now + 1900 + Math.random() * 3200;
      }
      tx = this.wander.x;
      ty = this.wander.y;
    }

    // While thinking, the gaze drifts toward a slowly wandering spot up and
    // to the side — the "hmm" look. Blended, so it eases in and out.
    const bt = this.stateBlend.thinking;
    if (bt > 0.01) {
      if (now >= this.think.nextAt) {
        this.think.x = w * (0.12 + Math.random() * 0.5);
        this.think.y = h * (0.12 + Math.random() * 0.22);
        this.think.nextAt = now + 1600 + Math.random() * 2200;
      }
      tx += (this.think.x - tx) * bt;
      ty += (this.think.y - ty) * bt;
    }

    this.gaze.x = tx;
    this.gaze.y = ty;
  }

  /**
   * Clamp a world-space gaze target to a natural range of eye motion AROUND
   * the eyeballs' actual position. Without this the eyes roll to whatever
   * extreme the cursor stream sends (the overlay sits at the screen edge, so
   * the pointer is usually far above/outside the window) — the irises wedge
   * under the upper lids, which reads as a stuck upward stare or a
   * half-closed eye. Returns the clamped target in `out` and records the
   * clamped yaw/pitch for the head-follow.
   */
  clampGaze(target, out) {
    const ex = this.eyeWorld.x;
    const ey = this.eyeWorld.y;
    // The eyes sit essentially on the model plane (z ~ 0).
    const dx = target.x - ex;
    const dy = target.y - ey;
    const dz = target.z;
    const dist = Math.max(0.25, Math.hypot(dx, dy, dz));

    let pitch = Math.atan2(dy, Math.hypot(dx, dz)); // + = up
    let yaw = Math.atan2(dx, Math.max(0.05, dz)); // + = her left
    pitch = Math.max(-EYE_PITCH_DOWN_MAX, Math.min(EYE_PITCH_UP_MAX, pitch));
    yaw = Math.max(-EYE_YAW_MAX, Math.min(EYE_YAW_MAX, yaw));
    this.lookPitch = pitch;
    this.lookYaw = yaw;

    const cp = Math.cos(pitch);
    return out.set(
      ex + Math.sin(yaw) * cp * dist,
      ey + Math.sin(pitch) * dist,
      Math.cos(yaw) * cp * dist,
    );
  }

  /**
   * Turn the head chain toward the (clamped) gaze with a damped lag. The
   * pitch range sits on top of the near-level neutral tilt and is clamped
   * asymmetrically: a small dip down is comfortable, looking up is limited
   * to a few degrees so the head never locks upward — and never bows.
   */
  updateHeadFollow(delta) {
    const targetYaw = Math.max(
      -HEAD_YAW_MAX,
      Math.min(HEAD_YAW_MAX, this.lookYaw * 0.55),
    );
    // VRM convention: pitch+ = down; lookPitch+ = up.
    const pitchDelta = Math.max(
      -HEAD_PITCH_UP_MAX,
      Math.min(HEAD_PITCH_DOWN_MAX, -this.lookPitch * HEAD_FOLLOW_FRACTION),
    );
    const targetPitch = NEUTRAL_HEAD_PITCH + pitchDelta;
    const k = 1 - Math.exp(-HEAD_FOLLOW_RATE * delta);
    this.headYaw += (targetYaw - this.headYaw) * k;
    this.headPitch += (targetPitch - this.headPitch) * k;
  }

  /* ------------------------------------------------------ idle actions --- */

  /** Smooth 0 -> 1 -> 0 envelope so actions ease in and out. Squared sine
   * gives zero velocity at both ends — movements start and settle like
   * butter instead of snapping in and out. */
  bump(t) {
    const s = Math.sin(Math.PI * t);
    return s * s;
  }

  /** Ease in, hold, ease out — for poses that linger briefly. */
  plateau(t) {
    return smoothstep(0, 0.3, t) * (1 - smoothstep(0.7, 1, t));
  }

  /**
   * Rotate a normalized humanoid bone by a yaw/pitch/roll euler using the
   * face-front conjugation (same math VRM's lookAt bone applier uses), so the
   * signs are correct for VRM 0.x and 1.0 alike. Convention: yaw+ = her left,
   * pitch+ = down, roll+ = tilt toward her right.
   */
  setBoneEuler(name, x = 0, y = 0, z = 0) {
    const node = this.bones[name];
    if (!node) return;
    _euler.set(x, y, z, 'YXZ');
    _quat.setFromEuler(_euler);
    node.quaternion.copy(this.faceFrontQuat).multiply(_quat).multiply(this.faceFrontQuatInv);
  }

  /**
   * Facial action lifecycle: random scheduled actions were removed by
   * request, so this channel only ever carries poke reactions — it merely
   * retires an action once its duration has elapsed.
   */
  updateAction(now) {
    if (!this.action) return;
    if ((now - this.action.t0) / this.action.duration >= 1) this.action = null;
  }

  /**
   * Whole-body action lifecycle: same story — the poke reaction (hop) is
   * the only source, so this only retires a finished action.
   */
  updateBodyAction(now) {
    if (!this.bodyAction) return;
    if ((now - this.bodyAction.t0) / this.bodyAction.duration >= 1) {
      this.bodyAction = null;
    }
  }

  /** One quick, light two-footed bounce — kept ONLY as a poke reaction
   * (jumps/bounces are otherwise disabled by request). */
  makeHop() {
    const unitH = this.fit.unitSize.height;
    const height = unitH * (0.018 + Math.random() * 0.012);
    return {
      duration: 850 + Math.random() * 300,
      blocksFace: true,
      apply: (t, pose) => {
        const tLaunch = 0.14; // little dip ends, pop begins
        const tLand = 0.58; // touchdown
        if (t < tLaunch) {
          const p = smoothstep(0, 1, t / tLaunch);
          pose.dy = -unitH * 0.012 * p;
          pose.kneeL = pose.kneeR = 0.22 * p;
        } else if (t < tLand) {
          const s = (t - tLaunch) / (tLand - tLaunch);
          const arc = 4 * s * (1 - s);
          const dip = 1 - smoothstep(0, 0.3, s);
          pose.dy = height * arc - unitH * 0.012 * dip;
          pose.kneeL = pose.kneeR = 0.22 * dip;
        } else {
          const p = 1 - smoothstep(0, 1, (t - tLand) / (1 - tLand));
          pose.dy = -unitH * 0.012 * p;
          pose.kneeL = pose.kneeR = 0.22 * p;
        }
      },
    };
  }

  /* ------------------------------------------------------- poke reaction --- */

  /**
   * React right away to a poke (a click on her): plays one of a few quick
   * responses, replacing whatever was playing.
   */
  poke() {
    const now = performance.now();
    const reactions = [];

    if (this.hasExpression('happy')) {
      // Giggle: happy squint with a little rock.
      reactions.push(() => ({
        duration: 1600 + Math.random() * 600,
        apply: (t, frame) => {
          const g = this.bump(t);
          frame.extraExpressions ??= {};
          frame.extraExpressions.happy = 0.85 * g;
          frame.blinkAdd = 0.4 * g;
          frame.bodyRoll = 0.05 * Math.sin(4 * Math.PI * t) * g;
        },
      }));
    }

    const surprised = this.findExpression(/surpris|びっくり|驚/i);
    if (surprised) {
      // Surprised double-take: brows up, head recoils a touch.
      reactions.push(() => ({
        duration: 1300 + Math.random() * 500,
        apply: (t, frame) => {
          const s = this.bump(t);
          frame.extraExpressions ??= {};
          frame.extraExpressions[surprised] = 0.8 * s;
          frame.headPitch = -0.1 * s;
          frame.gazeY = -window.innerHeight * 0.06 * s;
        },
      }));
    }

    // Curious head tilt with a playful BOTH-eyes squeeze (never a one-eye
    // wink — single-eye closures read as a broken blink).
    reactions.push(() => {
      const amp = Math.random() < 0.5 ? -0.3 : 0.3;
      return {
        duration: 1700 + Math.random() * 600,
        apply: (t, frame) => {
          frame.headRoll = amp * this.bump(t);
          frame.blinkAdd = 0.85 * this.plateau(t);
        },
      };
    });

    // Quick little hop.
    reactions.push(() => ({ kind: 'body', action: this.makeHop() }));

    const make = reactions[Math.floor(Math.random() * reactions.length)];
    const picked = make();
    // Snap guard: the interrupted action is not dropped in a single frame —
    // it fades out over POKE_FADE_MS while the reaction crossfades in.
    // The reaction MUST be stamped with t0 — without it the (now - t0)
    // progress math yields NaN, the reaction never renders, and the
    // finished check never fires, which wedges the whole action channel
    // for the rest of the session.
    if (picked.kind === 'body') {
      if (this.bodyAction) this.fadingBody = { action: this.bodyAction, t0: now };
      this.bodyAction = { t0: now, ...picked.action };
    } else {
      if (this.action) this.fadingFace = { action: this.action, t0: now };
      this.action = { t0: now, ...picked };
    }
  }

  /* -------------------------------------------------------- talking ------- */

  /**
   * Lecturer-style talking gestures (#6). While `speaking`, she moves like a
   * person explaining something: the arms work IN FRONT of her body with all
   * three joints articulating — shoulder lifts and brings the arm forward,
   * elbow bends so the hand works at chest height, and the WRIST bends and
   * the PALM rotates so the hand leads the gesture (open-palm "offering an
   * idea" shapes, soft cups between phrases) instead of swinging as one rigid
   * piece. Small head nods and tilts ride the speech rhythm, plus a light
   * body sway. Everything eases in/out with `talkBlend` and runs at a STEADY,
   * natural amplitude — it is deliberately NOT driven by the voice level, so
   * the arms, head and face don't pump up and down like a volume meter. Two
   * incommensurate sine layers per channel keep the flow from visibly
   * repeating.
   */
  updateTalking(delta, pose, frame) {
    // The voice arrives clip by clip (one per sentence, with synthesis gaps
    // between them), so the gesture state follows the CONVERSATION, not the
    // individual clip: it stays up for as long as the backend says she is
    // speaking, instead of slumping and swelling on every sentence gap like
    // a volume meter. The mouth (lip-sync) still follows the audio itself.
    const talkActive = this.speaking || this.backendState === 'speaking';
    // Fresh gesture flavor for each utterance: re-roll the seeds when the
    // conversation starts so no two talking stretches move alike (the phase
    // also jumps so the pattern never resumes where the last one left off).
    if (talkActive && !this._wasTalking) {
      this.talk.seedA = Math.random() * Math.PI * 2;
      this.talk.seedB = Math.random() * Math.PI * 2;
      this.talk.amp = 0.8 + Math.random() * 0.4;
      this.talk.phase += 1.5 + Math.random() * 2.5;
    }
    this._wasTalking = talkActive;

    // Ease the blend toward 1 while the conversation runs, back to 0 when
    // the backend returns to idle.
    const target = talkActive ? 1 : 0;
    const k = 1 - Math.exp(-(talkActive ? 3.0 : 2.0) * delta);
    this.talkBlend += (target - this.talkBlend) * k;
    if (this.talkBlend < 0.002 && !talkActive) {
      this.talkBlend = 0;
      return;
    }
    const blend = this.talkBlend;

    // Advance the gesture clock at a steady, natural tempo — deliberately
    // NOT tied to the voice level, so the movement doesn't speed up and down
    // with loudness. Frame-rate independent.
    this.talk.phase += delta * 1.1;
    const ph = this.talk.phase;
    const A = this.talk.seedA;
    const B = this.talk.seedB;
    const amp = this.talk.amp;

    // While a whole-body movement (the poke hop) owns the limbs, the arm
    // gestures step way back so they never stack on top of it — the head
    // and sway keep going.
    const busy = Boolean(this.bodyAction && this.bodyAction.blocksFace);
    const limb = blend * (busy ? 0.25 : 1);

    // Per-side gesture envelopes (0..1): rectified sine pairs so the hands
    // take turns instead of moving in lockstep. The two incommensurate sine
    // layers keep the flow organic at a steady amplitude.
    const gestL = Math.max(0, Math.sin(ph * 1.3 + A) * 0.6 + Math.sin(ph * 0.7 + B) * 0.4);
    const gestR = Math.max(0, Math.sin(ph * 1.1 + B + 1.9) * 0.6 + Math.sin(ph * 0.6 + A + 0.7) * 0.4);

    const gAmp = 0.75 * amp * limb;

    // SHOULDER: lift the gesturing arm and bring it in front of the body
    // (yaw+ = left arm back / right arm forward, hence opposite signs).
    pose.armL += gestL * gAmp * 0.75;
    pose.armR += gestR * gAmp * 0.85;
    pose.armSwingL -= gestL * 0.34 * limb;
    pose.armSwingR += gestR * 0.34 * limb;
    // ELBOW: bent so the hand works in front of the chest, never a straight
    // rigid arm.
    pose.elbowL += gestL * gAmp * 1.15;
    pose.elbowR += gestR * gAmp * 1.05;
    // WRIST: the hand leads — flex on the rise, relax on the fall, with a
    // continuous little bob so the wrist is never frozen.
    pose.wristL += (gestL * 0.32 + 0.1 * Math.sin(ph * 2.6 + A)) * limb;
    pose.wristR += (gestR * 0.32 + 0.1 * Math.sin(ph * 2.2 + B)) * limb;
    // PALM: rotate outward/up like offering an idea (mirrored per side).
    pose.palmL -= gestL * 0.3 * limb;
    pose.palmR += gestR * 0.3 * limb;
    // HAND: open palm while presenting (fingers extend), soft cup at rest.
    pose.handOpen += Math.max(gestL, gestR) * 0.6 * limb;

    // Small head nods/tilts riding the speech rhythm at a steady amplitude —
    // natural conversational nods, not punched by loudness.
    const nod = Math.sin(ph * 2.1 + A) * 0.5 + Math.sin(ph * 3.3 + B) * 0.3;
    frame.headPitch = (frame.headPitch || 0) + nod * 0.04 * blend;
    frame.headRoll = (frame.headRoll || 0) + Math.sin(ph * 0.9 + B) * 0.05 * blend;
    frame.headYaw = (frame.headYaw || 0) + Math.sin(ph * 0.7 + A) * 0.04 * blend;

    // Light body sway at a steady amplitude.
    pose.rot += Math.sin(ph * 0.8 + A) * 0.006 * blend;
    pose.dx +=
      this.fit.unitSize.width * 0.003 * Math.sin(ph * 0.6 + B) * blend;

    // No talking-driven facial expressions: a held smile squints the eyes and
    // a loudness-driven brow lift reads as weird pumping. Her face stays
    // natural while speaking — the backend emotion system supplies real
    // expressions per reply, and blinks + lip-sync keep the face alive.
  }

  /* -------------------------------------------------------- frame write --- */

  /**
   * Compose everything for this frame and write it to the model. Runs BEFORE
   * vrm.update(delta) in the render loop, so the humanoid copy, lookAt
   * compensation and expression application all see this frame's values.
   */
  applyFrame(delta) {
    const now = performance.now();

    // Degenerate viewport (minimize/restore race, DPI change): the gaze math
    // divides by the window size and would latch NaN into the damped head
    // state and the lookAt target — permanently. Hold the last good pose
    // until the real size is back; this is the same predicate fitModel uses.
    const winW = window.innerWidth;
    const winH = window.innerHeight;
    if (!winW || !winH || winW < 8 || winH < 8) return;

    // Retire any finished (poke-reaction) actions.
    this.updateStateBlend(delta);
    this.updateBodyAction(now);
    this.updateAction(now);

    // Whole-body pose bucket: neutral unless a body action is playing.
    const pose = {
      dx: 0,
      dy: 0,
      rot: 0,
      spinePitch: 0,
      kneeL: 0,
      kneeR: 0,
      armL: 0,
      armR: 0,
      armSwingL: 0,
      armSwingR: 0,
      elbowL: 0,
      elbowR: 0,
      wristL: 0,
      wristR: 0,
      palmL: 0,
      palmR: 0,
      handOpen: 0,
      shrugBreath: 0,
      armSway: 0,
    };

    // Always-on signs of life: a soft breathing bob with a hint of chest
    // rise, plus a slow micro sway, so she is never perfectly still. The
    // shoulders rise a hair on the inhale and the hanging arms lag the sway
    // slightly — breath and sway travel through the whole body, not just
    // the torso.
    const { width: unitW, height: unitH } = this.fit.unitSize;
    const breath = Math.sin(now * ((2 * Math.PI) / 4500));
    const sway = Math.sin(now * ((2 * Math.PI) / 7300) + 1.7);
    pose.dy += unitH * 0.0035 * breath;
    pose.spinePitch += 0.01 * breath;
    pose.dx += unitW * 0.004 * sway;
    pose.rot += 0.004 * sway;
    pose.shrugBreath += 0.012 * Math.max(0, breath);
    pose.armSway += 0.012 * sway;

    // Facial action bucket (created before the body action runs so big
    // movements can contribute expressions too).
    const frame = {};
    if (this.bodyAction) {
      const t = (now - this.bodyAction.t0) / this.bodyAction.duration;
      if (t < 1) this.bodyAction.apply(t, pose, frame);
    }
    if (this.action) {
      const t = (now - this.action.t0) / this.action.duration;
      if (t < 1) this.action.apply(t, frame);
    }

    // Poke snap guard: an action interrupted by a poke keeps contributing
    // while it decays, crossfaded with the reaction that replaced it.
    if (this.fadingBody) {
      const p = (now - this.fadingBody.t0) / POKE_FADE_MS;
      if (p >= 1) {
        this.fadingBody = null;
      } else {
        const a = this.fadingBody.action;
        const poseOld = {};
        const frameOld = {};
        a.apply(Math.min(1, (now - a.t0) / a.duration), poseOld, frameOld);
        mergeBucket(pose, poseOld, 1 - p);
        mergeBucket(frame, frameOld, 1 - p);
        if (frameOld.extraExpressions) {
          frame.extraExpressions ??= {};
          mergeBucket(frame.extraExpressions, frameOld.extraExpressions, 1 - p);
        }
      }
    }
    if (this.fadingFace) {
      const p = (now - this.fadingFace.t0) / POKE_FADE_MS;
      if (p >= 1) {
        this.fadingFace = null;
      } else {
        const a = this.fadingFace.action;
        const frameOld = {};
        a.apply(Math.min(1, (now - a.t0) / a.duration), frameOld);
        mergeBucket(frame, frameOld, 1 - p);
        if (frameOld.extraExpressions) {
          frame.extraExpressions ??= {};
          mergeBucket(frame.extraExpressions, frameOld.extraExpressions, 1 - p);
        }
      }
    }

    // Talking gestures layer on top of whatever else is playing.
    this.updateTalking(delta, pose, frame);

    // Assistant-state coloring: attentive lean-in with a hint of wide eyes
    // while listening; a slight head tilt while thinking. The composed head
    // pitch is hard-clamped later, so she can never lock looking upward.
    const bl = this.stateBlend.listening;
    if (bl > 0.004) {
      pose.spinePitch -= 0.05 * bl; // eager forward lean
      frame.headPitch = (frame.headPitch || 0) - 0.035 * bl; // chin up a touch
      // Slow little attentive nods — "I'm with you" — while listening.
      frame.headPitch +=
        0.022 * bl * (0.5 + 0.5 * Math.sin(now * ((2 * Math.PI) / 1500)));
      const surprised = this.findExpression(/surpris|びっくり|驚/i);
      if (surprised) {
        frame.extraExpressions ??= {};
        frame.extraExpressions[surprised] = Math.max(
          frame.extraExpressions[surprised] || 0,
          0.16 * bl,
        );
      }
    }
    const bt = this.stateBlend.thinking;
    if (bt > 0.004) {
      frame.headRoll = (frame.headRoll || 0) + 0.06 * bt;
      frame.headPitch = (frame.headPitch || 0) - 0.04 * bt;
    }

    // --- gaze: pick the target, map it to world space, clamp it to a
    // natural range of eye motion around the eyeballs, then let the head
    // follow the clamped direction (actions may offset the gaze — eye rolls,
    // glances, shy looks).
    this.updateGaze(now);
    this.fit.gazeToWorld(
      this.gaze.x,
      this.gaze.y + (frame.gazeY || 0),
      _gazeVec,
    );
    this.clampGaze(_gazeVec, _gazeVec);
    this.updateHeadFollow(delta);
    // Saccade smoothing: the eyes glide to the new target over ~40 ms
    // instead of teleporting, so wander hops, glances and cursor tracking
    // read as real eye movements.
    if (!this.gazePos) this.gazePos = _gazeVec.clone();
    this.gazePos.lerp(_gazeVec, 1 - Math.exp(-28 * delta));
    this.lookAtTarget.position.copy(this.gazePos);

    // --- expressions: clean slate first — three-vrm keeps a weight until it
    // is changed, so anything a finished action set would otherwise freeze on
    // her face forever (a stuck smile, a stuck squint...). Every frame writes
    // exactly what it wants on top of zero. Mouth visemes and action
    // expressions next; the blink is applied LAST so it is always the final,
    // authoritative word on the eyes.
    //
    // Conversation emotion (pushed by the backend with each reply) is held
    // for the duration of the utterance — kept alive while she speaks — and
    // fades out over the last ~1.2 s, so her face matches what she is saying.
    if (this.prevEmotion) {
      const p = (now - this.prevEmotion.t0) / EMOTION_CROSSFADE_MS;
      if (p >= 1) {
        this.prevEmotion = null;
      } else {
        frame.extraExpressions ??= {};
        for (const [name, weight] of Object.entries(this.prevEmotion.weights)) {
          if (!this.hasExpression(name)) continue;
          frame.extraExpressions[name] = Math.max(
            frame.extraExpressions[name] || 0,
            weight * (1 - p),
          );
        }
      }
    }
    if (this.emotionWeights && now < this.emotionUntil) {
      if (this.speaking) this.emotionUntil = Math.max(this.emotionUntil, now + 2500);
      const fade = Math.min(1, (this.emotionUntil - now) / 1200);
      const rampIn = Math.min(1, (now - this.emotionSetAt) / EMOTION_CROSSFADE_MS);
      // Show the emotion at full strength when it lands, then settle to a
      // softer sustain — holding the peak for a whole utterance freezes the
      // face (squinted eyes, pulled brows) while she talks.
      const settle =
        1 -
        (1 - EMOTION_SUSTAIN) *
          smoothstep(500, 500 + EMOTION_SETTLE_MS, now - this.emotionSetAt);
      frame.extraExpressions ??= {};
      for (const [name, weight] of Object.entries(this.emotionWeights)) {
        if (!this.hasExpression(name)) continue;
        frame.extraExpressions[name] = Math.max(
          frame.extraExpressions[name] || 0,
          weight * fade * rampIn * settle,
        );
      }
    }
    if (this.expressions) this.expressions.resetValues();
    this.updateMouth(delta);
    if (frame.extraExpressions) {
      for (const [name, weight] of Object.entries(frame.extraExpressions)) {
        this.setExpression(name, weight);
      }
    }
    this.updateBlink(now);
    this.applyBlink(frame.blinkAdd || 0);

    // --- bones: clean slate, resting pose, then head-follow + action layers.
    for (const node of Object.values(this.bones)) {
      node.quaternion.identity();
    }
    for (const side of ['left', 'right']) {
      for (const chain of this.fingers[side]) {
        for (const node of chain) node.quaternion.identity();
      }
    }

    // Resting pose: arms hang to her sides, elbows softly bent forward.
    // Elbow/wrist/finger channels add on top of the relaxed baseline. The
    // hanging arms carry a whisper of the body sway so they never read as
    // pinned to the torso.
    const elbowL = REST_ELBOW_YAW + (pose.elbowL || 0);
    const elbowR = REST_ELBOW_YAW + (pose.elbowR || 0);
    const armSway = pose.armSway || 0;
    if (this.bones.leftUpperArm) {
      this.setBoneEuler('leftUpperArm', 0, pose.armSwingL, -REST_ARM_ROLL + armSway);
    }
    if (this.bones.rightUpperArm) {
      this.setBoneEuler('rightUpperArm', 0, pose.armSwingR, REST_ARM_ROLL - armSway);
    }
    if (this.bones.leftLowerArm) this.setBoneEuler('leftLowerArm', 0, -elbowL, 0);
    if (this.bones.rightLowerArm) this.setBoneEuler('rightLowerArm', 0, elbowR, 0);

    // Hands: a soft wrist bend + relaxed finger curl at all times so they
    // never read as rigid paddles; gestures add wrist flex, palm rotation
    // (yaw — the hand leads the arm like a real presenter) and open the
    // fingers for open-palm shapes. The bend is a gentle forward cup
    // (negative pitch on these normalized bones).
    const wristL = REST_WRIST_PITCH + (pose.wristL || 0);
    const wristR = REST_WRIST_PITCH + (pose.wristR || 0);
    if (this.bones.leftHand) this.setBoneEuler('leftHand', -wristL, pose.palmL || 0, 0);
    if (this.bones.rightHand) this.setBoneEuler('rightHand', -wristR, pose.palmR || 0, 0);
    // handOpen extends the fingers toward a flat open palm (never hyper-
    // extending past straight).
    const curl = Math.max(-0.05, REST_FINGER_CURL - 0.22 * (pose.handOpen || 0));
    this.applyFingerCurl(curl);

    // Head chain: damped gaze follow + action offsets, split across bones.
    // Hard clamps guarantee the composed pose can never lock upward (or bow)
    // no matter which action layers are playing. Two slow incommensurate
    // micro drifts keep the head from ever freezing between actions —
    // perfect stillness is the one thing that reads as a mannequin.
    const microYaw = 0.008 * Math.sin(now * ((2 * Math.PI) / 9700) + 0.5);
    const microPitch = 0.006 * Math.sin(now * ((2 * Math.PI) / 11300) + 2.1);
    const yaw = Math.max(
      -HEAD_YAW_HARD_MAX,
      Math.min(HEAD_YAW_HARD_MAX, this.headYaw + (frame.headYaw || 0) + microYaw),
    );
    const pitch = Math.max(
      HEAD_PITCH_HARD_MIN,
      Math.min(HEAD_PITCH_HARD_MAX, this.headPitch + (frame.headPitch || 0) + microPitch),
    );
    const roll = frame.headRoll || 0;
    if (this.bones.head) {
      this.setBoneEuler('head', pitch * HEAD_SPLIT.head, yaw * HEAD_SPLIT.head, roll * 0.7);
    }
    if (this.bones.neck) {
      this.setBoneEuler('neck', pitch * HEAD_SPLIT.neck, yaw * HEAD_SPLIT.neck, roll * 0.3);
    }
    if (this.bones.spine) {
      this.setBoneEuler(
        'spine',
        pitch * HEAD_SPLIT.spine + pose.spinePitch,
        yaw * HEAD_SPLIT.spine,
        0,
      );
    }

    // Action limbs layer on top of the resting pose (facial actions and
    // whole-body movements can both raise the arms).
    const armLiftL = pose.armL || 0;
    const armLiftR = pose.armR || 0;
    if (armLiftL) {
      this.setBoneEuler(
        'leftUpperArm',
        0,
        pose.armSwingL,
        -REST_ARM_ROLL + armLiftL + armSway,
      );
    }
    if (armLiftR) {
      this.setBoneEuler(
        'rightUpperArm',
        0,
        pose.armSwingR,
        REST_ARM_ROLL - armLiftR - armSway,
      );
    }
    // Shoulders carry the breathing lift.
    if (pose.shrugBreath) {
      this.setBoneEuler('leftShoulder', 0, 0, pose.shrugBreath);
      this.setBoneEuler('rightShoulder', 0, 0, -pose.shrugBreath);
    }
    // Legs: pitch+ on the lower leg bends the knee (heel coming up behind)
    // — the little hop action uses it.
    if (pose.kneeL) this.setBoneEuler('leftLowerLeg', pose.kneeL, 0, 0);
    if (pose.kneeR) this.setBoneEuler('rightLowerLeg', pose.kneeR, 0, 0);

    // --- whole-body transform, pivoted at the feet. pose.rot tilts in the
    // screen plane.
    this.body.position.set(pose.dx, pose.dy, 0);
    this.body.rotation.set(0, 0, pose.rot + (frame.bodyRoll || 0));
  }

  /**
   * Curl every finger chain by `amount` (radians, split across the three
   * joints with the thumb gentler). Applied as a negative pitch — a gentle
   * forward cup on these normalized bones — using the same face-front
   * conjugation as setBoneEuler so the sign is correct for VRM 0.x and 1.0.
   */
  applyFingerCurl(amount) {
    if (!amount) return;
    for (const side of ['left', 'right']) {
      for (let f = 0; f < this.fingers[side].length; f++) {
        const chain = this.fingers[side][f];
        const isThumb = f === 0;
        for (let j = 0; j < chain.length; j++) {
          // Proximal joints do most of the curl; distal joints a little less.
          const w = j === 0 ? 1 : j === 1 ? 0.85 : 0.55;
          const a = amount * w * (isThumb ? 0.45 : 1);
          _euler.set(-a, 0, 0, 'YXZ');
          _quat.setFromEuler(_euler);
          chain[j].quaternion
            .copy(this.faceFrontQuat)
            .multiply(_quat)
            .multiply(this.faceFrontQuatInv);
        }
      }
    }
  }

  destroy() {
    window.removeEventListener('mousemove', this._onMouseMove);
    if (this.vrm.lookAt && this.vrm.lookAt.target === this.lookAtTarget) {
      this.vrm.lookAt.target = null;
    }
  }
}
