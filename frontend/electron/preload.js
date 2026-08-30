const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('overlay', {
  /** Toggle click-through for the whole window. */
  setIgnoreMouse: (ignore) => ipcRenderer.send('overlay:set-ignore', ignore),

  /** Dragging: grab offset is window-relative; moves use screen coords. */
  dragStart: (grabX, grabY) => ipcRenderer.send('overlay:drag-start', grabX, grabY),
  dragMove: (screenX, screenY) => ipcRenderer.send('overlay:drag-move', screenX, screenY),
  dragEnd: () => ipcRenderer.send('overlay:drag-end'),

  togglePin: () => ipcRenderer.invoke('overlay:toggle-pin'),
  toggleTrackMouse: () => ipcRenderer.invoke('overlay:toggle-track-mouse'),
  /** Voice replies (TTS) on/off — also told to the backend pipeline. */
  toggleVoice: () => ipcRenderer.invoke('overlay:toggle-voice'),
  getState: () => ipcRenderer.invoke('overlay:get-state'),
  setModel: (rel) => ipcRenderer.invoke('overlay:set-model', rel),

  /** Container edit modes: 'off' | 'container' | 'character'. */
  setEditMode: (mode) => ipcRenderer.invoke('overlay:set-edit-mode', mode),
  /** Resize/move the overlay window itself (bounds in screen coords) — the
   * window is the container the character is fitted into. */
  setWindowBounds: (bounds) => ipcRenderer.send('overlay:set-window-bounds', bounds),
  /** Persist the character offset (fractions of the window). */
  setCharOffset: (charOffset) => ipcRenderer.send('overlay:set-char-offset', charOffset),

  close: () => ipcRenderer.send('overlay:close'),

  /** Renderer finished booting (model loaded) — the main process starts the
   * voice backend only after this so no events are lost. */
  ready: () => ipcRenderer.send('overlay:ready'),

  /** The current TTS clip has finished playing (or was stopped). Forwarded
   * to the voice backend, which keeps the mic gated until this lands — the
   * echo guard that stops her hearing her own voice. The id echoes the clip
   * the confirmation belongs to, so a stale ack can never release the gate
   * for a newer clip. */
  ttsDone: (id) => ipcRenderer.send('overlay:tts-done', id == null ? null : id),

  onStateChanged: (cb) => {
    const listener = (_e, payload) => cb(payload);
    ipcRenderer.on('overlay:state-changed', listener);
    return () => ipcRenderer.removeListener('overlay:state-changed', listener);
  },
  onModelChanged: (cb) => {
    const listener = (_e, payload) => cb(payload);
    ipcRenderer.on('overlay:model-changed', listener);
    return () => ipcRenderer.removeListener('overlay:model-changed', listener);
  },

  /** Global cursor position (window-relative), streamed so the gaze can follow
   * the pointer even outside the window. */
  onCursor: (cb) => {
    const listener = (_e, payload) => cb(payload);
    ipcRenderer.on('overlay:cursor', listener);
    return () => ipcRenderer.removeListener('overlay:cursor', listener);
  },

  /**
   * Events pushed from the assistant.py voice pipeline (spawned and bridged
   * by the main process):
   * { type: 'state', value: 'idle'|'listening'|'thinking'|'speaking' }
   * { type: 'emotion', value: 'happy'|'smug'|'evil'|'angry'|'sad'|'surprised'|'neutral' }
   * { type: 'speak', value: { id: <clip id>, wav: '<wav data: URI>' } }
   */
  onBackendEvent: (cb) => {
    const listener = (_e, payload) => cb(payload);
    ipcRenderer.on('backend:event', listener);
    return () => ipcRenderer.removeListener('backend:event', listener);
  },
});
