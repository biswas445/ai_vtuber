/**
 * Launches the Electron app with ELECTRON_RUN_AS_NODE removed from the
 * environment.
 *
 * Electron treats a *present* ELECTRON_RUN_AS_NODE variable (even an empty
 * one) as "run as plain Node", which makes `require('electron')` return the
 * executable path string instead of the API object. Some parent shells set
 * it, and tools like cross-env can only blank it, not delete it — so we strip
 * it here and spawn the real binary ourselves.
 *
 * Run from plain Node: `require('electron')` resolves to the Electron
 * executable path (see node_modules/electron/index.js).
 */

const { spawn } = require('node:child_process');
const path = require('node:path');

const electronExe = require('electron');

const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;

// The app root is the frontend directory (this file lives in electron/).
const appPath = path.join(__dirname, '..');

const child = spawn(electronExe, [appPath, ...process.argv.slice(2)], {
  stdio: 'inherit',
  env,
});

child.on('close', (code, signal) => {
  if (code !== null) process.exit(code);
  process.exit(signal ? 1 : 0);
});
