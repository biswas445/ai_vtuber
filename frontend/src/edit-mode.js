/**
 * Edit modes for the character's stage. The overlay window itself IS the
 * container the character is fitted into; these modes make its bounds
 * visible and directly manipulable:
 *
 * - 'container': outline + drag handles around the window. Drag a
 *   corner/edge to resize the window, drag the inside to move it — exactly
 *   like grabbing a normal app window's edges, and the character re-fits
 *   live into whatever size she is given.
 * - 'character': outline only. Drag anywhere inside to move the character
 *   around relative to the window (her feet anchor stays on screen).
 *
 * Window changes are computed in SCREEN coordinates: the window itself
 * moves/resizes under the pointer mid-drag, so window-relative coords would
 * chase their own tail (dragging the left edge would freeze the moment the
 * window starts following the cursor). The main process clamps the result
 * to the display and persists it.
 */

export function setupEditMode({
  getRect,
  getWindowBounds,
  onWindowBoundsChanged,
  getCharOffset,
  clampCharOffset = (off) => off,
  onCharOffsetChanged,
}) {
  const root = document.getElementById('container-edit');
  const hint = root.querySelector('.ce-hint');

  // Smallest the window may shrink to (px) — must match MIN_WIN_W/MIN_WIN_H
  // on the BrowserWindow in the main process.
  const MIN_W = 160;
  const MIN_H = 200;
  // Pointer slack around the rect that still counts as "over" it, so the
  // thin handles are easy to grab even though input is click-through
  // everywhere else.
  const HANDLE_MARGIN = 10;

  let mode = 'off'; // 'off' | 'container' | 'character'
  let drag = null;

  function setMode(next) {
    mode = next;
    drag = null;
    root.classList.toggle('hidden', next === 'off');
    root.classList.toggle('mode-container', next === 'container');
    root.classList.toggle('mode-character', next === 'character');
    root.classList.remove('dragging');
    refresh();
  }

  /** Sync the outline element with the current window rect. */
  function refresh() {
    if (mode === 'off') return;
    const r = getRect();
    root.style.left = `${r.x}px`;
    root.style.top = `${r.y}px`;
    root.style.width = `${r.w}px`;
    root.style.height = `${r.h}px`;
  }

  root.addEventListener('pointerdown', (e) => {
    if (mode === 'off' || e.button !== 0) return;
    const handle =
      mode === 'container' && e.target instanceof Element ? e.target.dataset.handle : undefined;
    drag = {
      // A grab on a handle resizes the window; anywhere else moves the
      // window ('container' mode) or the character ('character' mode).
      type: handle ? 'resize' : mode === 'container' ? 'move-window' : 'move-character',
      handle: handle || null,
      // Character drags are window-relative (the window stands still).
      startX: e.clientX,
      startY: e.clientY,
      // Window drags are screen-relative (the window moves under the
      // pointer), starting from the window's bounds at the grab.
      startScreenX: e.screenX,
      startScreenY: e.screenY,
      startBounds: getWindowBounds(),
      startOffset: getCharOffset(),
    };
    if (e.target instanceof Element && e.target.setPointerCapture) {
      try {
        e.target.setPointerCapture(e.pointerId);
      } catch {
        /* capture is best-effort */
      }
    }
    root.classList.add('dragging');
    e.preventDefault();
    e.stopPropagation();
  });

  window.addEventListener('pointermove', (e) => {
    if (!drag) return;

    if (drag.type === 'move-character') {
      // Move the character's feet anchor. The limits come from the renderer's
      // fit (clampCharOffset) — exactly what the rendering clamps with, so
      // the drag is symmetric: she follows the cursor 1:1 everywhere and
      // stops precisely where the visibility clamp stops her (she may hang
      // off any window edge, but a visible portion always stays in view).
      const dx = e.clientX - drag.startX;
      const dy = e.clientY - drag.startY;
      const next = clampCharOffset({
        x: drag.startOffset.x + dx,
        y: drag.startOffset.y + dy,
      });
      hint.textContent = `${Math.round(next.x)}, ${Math.round(next.y)}`;
      onCharOffsetChanged(next);
      return;
    }

    const dx = e.screenX - drag.startScreenX;
    const dy = e.screenY - drag.startScreenY;
    const s = drag.startBounds;
    let next;
    if (drag.type === 'move-window') {
      next = { x: s.x + dx, y: s.y + dy, w: s.w, h: s.h };
    } else {
      // Resize: edges named in the handle move; opposite edges stay put.
      let left = s.x;
      let top = s.y;
      let right = s.x + s.w;
      let bottom = s.y + s.h;
      if (drag.handle.includes('w')) left = s.x + dx;
      if (drag.handle.includes('e')) right = s.x + s.w + dx;
      if (drag.handle.includes('n')) top = s.y + dy;
      if (drag.handle.includes('s')) bottom = s.y + s.h + dy;
      if (drag.handle.includes('w') && right - left < MIN_W) left = right - MIN_W;
      if (drag.handle.includes('e') && right - left < MIN_W) right = left + MIN_W;
      if (drag.handle.includes('n') && bottom - top < MIN_H) top = bottom - MIN_H;
      if (drag.handle.includes('s') && bottom - top < MIN_H) bottom = top + MIN_H;
      next = { x: left, y: top, w: right - left, h: bottom - top };
    }
    hint.textContent = `${Math.round(next.w)} \u00d7 ${Math.round(next.h)}`;
    // The main process clamps to the display and applies the bounds; the
    // resize event that follows re-fits the character and re-syncs the
    // outline, so the box always shows what the window actually is.
    onWindowBoundsChanged(next);
  });

  const endDrag = () => {
    if (!drag) return;
    drag = null;
    root.classList.remove('dragging');
  };
  window.addEventListener('pointerup', endDrag);
  window.addEventListener('pointercancel', endDrag);
  // Losing focus mid-drag (Alt+Tab, UAC prompt) means pointerup never
  // arrives; without this the stale drag keeps isOver() true forever and
  // the window permanently swallows clicks meant for apps behind.
  window.addEventListener('blur', endDrag);

  /** Hit-test for the click-through gating (window-relative coords). */
  function isOver(x, y) {
    if (mode === 'off') return false;
    if (drag) return true; // never drop input mid-drag
    const r = getRect();
    return (
      x >= r.x - HANDLE_MARGIN &&
      x <= r.x + r.w + HANDLE_MARGIN &&
      y >= r.y - HANDLE_MARGIN &&
      y <= r.y + r.h + HANDLE_MARGIN
    );
  }

  return { setMode, isOver, refresh };
}
