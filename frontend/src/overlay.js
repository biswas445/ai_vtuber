/**
 * Overlay interaction: click-through toggling, window dragging, edge
 * resize grips, and a toolbar that is summoned by right-clicking the character.
 *
 * The window is click-through by default so clicks on empty space fall through
 * to whatever app is underneath. Electron still forwards mousemove events while
 * click-through (`setIgnoreMouseEvents(true, { forward: true })`), so we can
 * hit-test the character and reclaim input while the pointer is over her — or
 * over the thin window-edge grips, which is what lets the frameless window be
 * resized by dragging its edges.
 */

export function setupOverlay({
  getCharacterRect,
  overEditUI,
  onCycleModel,
  onPoke,
  onScale,
}) {
  const api = window.overlay;
  const controlsEl = document.getElementById('controls');
  const pinBtn = document.getElementById('btn-pin');
  const editBtn = document.getElementById('btn-edit');
  const moveBtn = document.getElementById('btn-move');

  // How long the toolbar stays visible after the pointer leaves it, so the
  // pointer can travel across the click-through gap between the character and
  // the toolbar without it disappearing mid-way.
  const CONTROLS_HIDE_DELAY_MS = 650;
  // Clicks/dblclicks arriving this soon after a real drag are drag artifacts.
  const POST_DRAG_SUPPRESS_MS = 400;
  // Pointer jitter below this (px) does not count as a drag move.
  const DRAG_MOVE_THRESHOLD = 3;
  // Width of the invisible grip along each window edge (px). The overlay is
  // click-through everywhere else, so input must be reclaimed here or the user
  // could never grab the frameless window's edges to resize/maximize it.
  const RESIZE_BORDER = 6;

  let ignoring = true;
  let dragging = false;
  let dragMoved = false;
  let dragStartX = 0;
  let dragStartY = 0;
  let lastDragEndAt = 0;
  let controlsVisible = false;
  let hideTimer = null;
  let lastX = -1;
  let lastY = -1;

  function setControlsVisible(visible) {
    if (controlsVisible === visible) return;
    controlsVisible = visible;
    controlsEl.classList.toggle('hidden', !visible);
  }

  function cancelControlsHide() {
    clearTimeout(hideTimer);
    hideTimer = null;
  }

  function scheduleControlsHide() {
    if (hideTimer !== null) return;
    hideTimer = setTimeout(() => {
      hideTimer = null;
      // mousemove is forwarded even while click-through, so the pointer may
      // have moved during the grace period — re-check before hiding.
      if (!overInteractive(lastX, lastY)) setControlsVisible(false);
    }, CONTROLS_HIDE_DELAY_MS);
  }

  function overControls(x, y) {
    if (!controlsVisible) return false;
    const r = controlsEl.getBoundingClientRect();
    return x >= r.left - 4 && x <= r.right + 4 && y >= r.top - 4 && y <= r.bottom + 4;
  }

  function overCharacter(x, y) {
    const r = getCharacterRect();
    return Boolean(r && x >= r.x && x <= r.x + r.width && y >= r.y && y <= r.y + r.height);
  }

  function overResizeBorder(x, y) {
    return (
      x <= RESIZE_BORDER ||
      y <= RESIZE_BORDER ||
      x >= window.innerWidth - RESIZE_BORDER ||
      y >= window.innerHeight - RESIZE_BORDER
    );
  }

  function overInteractive(x, y) {
    return overCharacter(x, y) || overControls(x, y) || overResizeBorder(x, y) || overEditUI(x, y);
  }

  window.addEventListener('mousemove', (e) => {
    if (dragging) return;
    lastX = e.clientX;
    lastY = e.clientY;
    const over = overInteractive(e.clientX, e.clientY);
    if (over && ignoring) {
      ignoring = false;
      api.setIgnoreMouse(false);
    } else if (!over && !ignoring) {
      ignoring = true;
      api.setIgnoreMouse(true);
    }
    // The toolbar is summoned by right-click only — hovering just keeps an
    // already-visible toolbar alive, and moving away lets it fade out.
    if (over) cancelControlsHide();
    else if (controlsVisible) scheduleControlsHide();
  });

  // Right-click the character (or anywhere while an edit mode is on) to
  // toggle the toolbar.
  window.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    if (e.target instanceof Element && e.target.closest('#controls')) return;
    if (!overCharacter(e.clientX, e.clientY) && !overEditUI(e.clientX, e.clientY)) return;
    if (controlsVisible) {
      cancelControlsHide();
      setControlsVisible(false);
    } else {
      setControlsVisible(true);
    }
  });

  window.addEventListener('pointerdown', (e) => {
    if (ignoring || e.button !== 0) return;
    if (e.target instanceof Element && e.target.closest('#controls')) return;
    // An edit mode owns the pointer inside the window (move/resize).
    if (overEditUI(e.clientX, e.clientY)) return;
    if (!overCharacter(e.clientX, e.clientY)) return;
    dragging = true;
    dragMoved = false;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    // Keep receiving pointer events even if the pointer outruns the window
    // while it chases the cursor (fast flicks), so the drag never stalls.
    if (e.target instanceof Element && e.target.setPointerCapture) {
      try {
        e.target.setPointerCapture(e.pointerId);
      } catch {
        /* capture is best-effort */
      }
    }
    api.dragStart(e.clientX, e.clientY);
    e.preventDefault();
  });

  window.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    if (
      !dragMoved &&
      Math.abs(e.clientX - dragStartX) + Math.abs(e.clientY - dragStartY) > DRAG_MOVE_THRESHOLD
    ) {
      dragMoved = true;
    }
    api.dragMove(e.screenX, e.screenY);
  });

  const endDrag = (e) => {
    if (!dragging) return;
    dragging = false;
    if (dragMoved) lastDragEndAt = performance.now();
    dragMoved = false;
    if (e && Number.isFinite(e.clientX)) {
      lastX = e.clientX;
      lastY = e.clientY;
    }
    api.dragEnd();
    // The pointer may have been released over empty space: re-evaluate
    // click-through right away, or the next motionless click would be
    // swallowed by the overlay instead of passing through to the app below.
    if (!overInteractive(lastX, lastY)) {
      ignoring = true;
      api.setIgnoreMouse(true);
    }
  };
  window.addEventListener('pointerup', endDrag);
  window.addEventListener('pointercancel', endDrag);
  window.addEventListener('blur', endDrag);

  // A drag ends with a click (and a fast second drag with a dblclick), which
  // would unintentionally cycle the model — swallow those synthetic events.
  const suppressPostDragEvents = (e) => {
    if (performance.now() - lastDragEndAt >= POST_DRAG_SUPPRESS_MS) return;
    e.stopImmediatePropagation();
    e.preventDefault();
  };
  window.addEventListener('click', suppressPostDragEvents, true);
  window.addEventListener('dblclick', suppressPostDragEvents, true);

  window.addEventListener('dblclick', (e) => {
    if (ignoring) return;
    if (e.target instanceof Element && e.target.closest('#controls')) return;
    if (overEditUI(e.clientX, e.clientY)) return;
    if (overCharacter(e.clientX, e.clientY)) onCycleModel();
  });

  // A plain click (no drag) on the character is a poke — she reacts.
  window.addEventListener('click', (e) => {
    if (ignoring) return;
    if (e.target instanceof Element && e.target.closest('#controls')) return;
    if (overEditUI(e.clientX, e.clientY)) return;
    if (!overCharacter(e.clientX, e.clientY)) return;
    onPoke();
  });

  pinBtn.addEventListener('click', async () => {
    const pinned = await api.togglePin();
    pinBtn.classList.toggle('active', pinned);
  });
  const trackBtn = document.getElementById('btn-track');
  trackBtn.addEventListener('click', async () => {
    const enabled = await api.toggleTrackMouse();
    trackBtn.classList.toggle('active', enabled);
  });
  const voiceBtn = document.getElementById('btn-voice');
  voiceBtn.addEventListener('click', async () => {
    const enabled = await api.toggleVoice();
    voiceBtn.classList.toggle('active', enabled);
  });
  document.getElementById('btn-model').addEventListener('click', onCycleModel);
  document.getElementById('btn-bigger').addEventListener('click', () => onScale(1.15));
  document.getElementById('btn-smaller').addEventListener('click', () => onScale(1 / 1.15));
  document.getElementById('btn-close').addEventListener('click', () => api.close());

  // Window-resize/character edit modes are mutually exclusive; clicking the
  // active button turns the mode back off.
  function syncEditButtons(mode) {
    editBtn.classList.toggle('active', mode === 'container');
    moveBtn.classList.toggle('active', mode === 'character');
  }
  editBtn.addEventListener('click', async () => {
    const next = editBtn.classList.contains('active') ? 'off' : 'container';
    syncEditButtons(await api.setEditMode(next));
  });
  moveBtn.addEventListener('click', async () => {
    const next = moveBtn.classList.contains('active') ? 'off' : 'character';
    syncEditButtons(await api.setEditMode(next));
  });
  // Drop keyboard focus after any toolbar click so a stray Enter/Space while
  // hovering the character cannot re-trigger the last button.
  controlsEl.addEventListener('click', (e) => {
    if (e.target instanceof Element) e.target.closest('button')?.blur();
  });

  api.getState().then(({ pinned, trackMouse, editMode, voice }) => {
    pinBtn.classList.toggle('active', pinned);
    trackBtn.classList.toggle('active', trackMouse !== false);
    voiceBtn.classList.toggle('active', voice !== false);
    syncEditButtons(editMode || 'off');
  });
  // Keep the toggle buttons in sync when their state changes from the tray.
  api.onStateChanged(({ pinned, trackMouse, editMode, voice }) => {
    pinBtn.classList.toggle('active', Boolean(pinned));
    if (typeof trackMouse === 'boolean') trackBtn.classList.toggle('active', trackMouse);
    if (typeof voice === 'boolean') voiceBtn.classList.toggle('active', voice);
    if (typeof editMode === 'string') syncEditButtons(editMode);
  });
}
