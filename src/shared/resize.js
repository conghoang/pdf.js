const MathLib = require("pica/lib/mathlib");

const __mathlib = new MathLib(["wasm", "js"]);

const __initMath = __mathlib.init();

export function initMath() {
  return __initMath;
}

export function resizeAndUnsharp(tileOpts) {
  return __mathlib.resizeAndUnsharp(tileOpts, {});
}
