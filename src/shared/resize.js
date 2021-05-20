"use strict";
import { blurMono16 } from "./glur.js";
const FILTER_INFO = [
  {
    // Nearest neibor (Box)
    win: 0.5,
    filter(x) {
      return x >= -0.5 && x < 0.5 ? 1.0 : 0.0;
    },
  },
  {
    // Hamming
    win: 1.0,
    filter(x) {
      if (x <= -1.0 || x >= 1.0) {
        return 0.0;
      }
      if (x > -1.1920929e-7 && x < 1.1920929e-7) {
        return 1.0;
      }
      const xpi = x * Math.PI;
      return (Math.sin(xpi) / xpi) * (0.54 + 0.46 * Math.cos(xpi / 1.0));
    },
  },
  {
    // Lanczos, win = 2
    win: 2.0,
    filter(x) {
      if (x <= -2.0 || x >= 2.0) {
        return 0.0;
      }
      if (x > -1.1920929e-7 && x < 1.1920929e-7) {
        return 1.0;
      }
      const xpi = x * Math.PI;
      return ((Math.sin(xpi) / xpi) * Math.sin(xpi / 2.0)) / (xpi / 2.0);
    },
  },
  {
    // Lanczos, win = 3
    win: 3.0,
    filter(x) {
      if (x <= -3.0 || x >= 3.0) {
        return 0.0;
      }
      if (x > -1.1920929e-7 && x < 1.1920929e-7) {
        return 1.0;
      }
      const xpi = x * Math.PI;
      return ((Math.sin(xpi) / xpi) * Math.sin(xpi / 3.0)) / (xpi / 3.0);
    },
  },
  {
    // Lanczos, win = 4
    win: 4.0,
    filter(x) {
      if (x <= -4.0 || x >= 4.0) {
        return 0.0;
      }
      if (x > -1.1920929e-7 && x < 1.1920929e-7) {
        return 1.0;
      }
      const xpi = x * Math.PI;
      return ((Math.sin(xpi) / xpi) * Math.sin(xpi / 4.0)) / (xpi / 4.0);
    },
  },
];

// Precision of fixed FP values
const FIXED_FRAC_BITS = 14;

function toFixedPoint(num) {
  return Math.round(num * ((1 << FIXED_FRAC_BITS) - 1));
}

const createFilters = function (quality, srcSize, destSize, scale, offset) {
  const filterFunction = FILTER_INFO[quality].filter;

  const scaleInverted = 1.0 / scale;
  const scaleClamped = Math.min(1.0, scale); // For upscale

  // Filter window (averaging interval), scaled to src image
  const srcWindow = FILTER_INFO[quality].win / scaleClamped;

  let destPixel,
    srcPixel,
    srcFirst,
    srcLast,
    filterElementSize,
    floatFilter,
    fxpFilter,
    total,
    pxl,
    idx,
    floatVal,
    filterTotal,
    filterVal;
  let leftNotEmpty, rightNotEmpty, filterShift, filterSize;

  const maxFilterElementSize = Math.floor((srcWindow + 1) * 2);
  const packedFilter = new Int16Array((maxFilterElementSize + 2) * destSize);
  let packedFilterPtr = 0;

  const slowCopy = !packedFilter.subarray || !packedFilter.set;

  // For each destination pixel calculate source range and built filter values
  for (destPixel = 0; destPixel < destSize; destPixel++) {
    // Scaling should be done relative to central pixel point
    srcPixel = (destPixel + 0.5) * scaleInverted + offset;

    srcFirst = Math.max(0, Math.floor(srcPixel - srcWindow));
    srcLast = Math.min(srcSize - 1, Math.ceil(srcPixel + srcWindow));

    filterElementSize = srcLast - srcFirst + 1;
    floatFilter = new Float32Array(filterElementSize);
    fxpFilter = new Int16Array(filterElementSize);

    total = 0.0;

    // Fill filter values for calculated range
    for (pxl = srcFirst, idx = 0; pxl <= srcLast; pxl++, idx++) {
      floatVal = filterFunction((pxl + 0.5 - srcPixel) * scaleClamped);
      total += floatVal;
      floatFilter[idx] = floatVal;
    }

    // Normalize filter, convert to fixed point and accumulate conversion error
    filterTotal = 0;

    for (idx = 0; idx < floatFilter.length; idx++) {
      filterVal = floatFilter[idx] / total;
      filterTotal += filterVal;
      fxpFilter[idx] = toFixedPoint(filterVal);
    }

    // Compensate normalization error, to minimize brightness drift
    fxpFilter[destSize >> 1] += toFixedPoint(1.0 - filterTotal);

    //
    // Now pack filter to useable form
    //
    // 1. Trim heading and tailing zero values, and compensate shitf/length
    // 2. Put all to single array in this format:
    //
    //    [ pos shift, data length, value1, value2, value3, ... ]
    //

    leftNotEmpty = 0;
    while (leftNotEmpty < fxpFilter.length && fxpFilter[leftNotEmpty] === 0) {
      leftNotEmpty++;
    }

    if (leftNotEmpty < fxpFilter.length) {
      rightNotEmpty = fxpFilter.length - 1;
      while (rightNotEmpty > 0 && fxpFilter[rightNotEmpty] === 0) {
        rightNotEmpty--;
      }

      filterShift = srcFirst + leftNotEmpty;
      filterSize = rightNotEmpty - leftNotEmpty + 1;

      packedFilter[packedFilterPtr++] = filterShift; // shift
      packedFilter[packedFilterPtr++] = filterSize; // size

      if (!slowCopy) {
        packedFilter.set(
          fxpFilter.subarray(leftNotEmpty, rightNotEmpty + 1),
          packedFilterPtr
        );
        packedFilterPtr += filterSize;
      } else {
        // fallback for old IE < 11, without subarray/set methods
        for (idx = leftNotEmpty; idx <= rightNotEmpty; idx++) {
          packedFilter[packedFilterPtr++] = fxpFilter[idx];
        }
      }
    } else {
      // zero data, write header only
      packedFilter[packedFilterPtr++] = 0; // shift
      packedFilter[packedFilterPtr++] = 0; // size
    }
  }
  return packedFilter;
};
// Convolve image in horizontal directions and transpose output. In theory,
// transpose allow:
//
// - use the same convolver for both passes (this fails due different
//   types of input array and temporary buffer)
// - making vertical pass by horisonltal lines inprove CPU cache use.
//
// But in real life this doesn't work :)
//
function convolveHorizontally(src, dest, srcW, srcH, destW, filters) {
  let r, g, b, a;
  let filterPtr, filterShift, filterSize;
  let srcPtr, srcY, destX, filterVal;
  let srcOffset = 0,
    destOffset = 0;

  // For each row
  for (srcY = 0; srcY < srcH; srcY++) {
    filterPtr = 0;

    // Apply precomputed filters to each destination row point
    for (destX = 0; destX < destW; destX++) {
      // Get the filter that determines the current output pixel.
      filterShift = filters[filterPtr++];
      filterSize = filters[filterPtr++];

      srcPtr = (srcOffset + filterShift * 4) | 0;

      r = g = b = a = 0;

      // Apply the filter to the row to get the destination pixel r, g, b, a
      for (; filterSize > 0; filterSize--) {
        filterVal = filters[filterPtr++];

        // Use reverse order to workaround deopts in old v8 (node v.10)
        // Big thanks to @mraleph (Vyacheslav Egorov) for the tip.
        a = (a + filterVal * src[srcPtr + 3]) | 0;
        b = (b + filterVal * src[srcPtr + 2]) | 0;
        g = (g + filterVal * src[srcPtr + 1]) | 0;
        r = (r + filterVal * src[srcPtr]) | 0;
        srcPtr = (srcPtr + 4) | 0;
      }

      // Bring this value back in range. All of the filter scaling factors
      // are in fixed point with FIXED_FRAC_BITS bits of fractional part.
      //
      // (!) Add 1/2 of value before clamping to get proper rounding. In other
      // case brightness loss will be noticeable if you resize image with white
      // border and place it on white background.
      //
      dest[destOffset + 3] = (a + (1 << 13)) >> 14; /* FIXED_FRAC_BITS */
      dest[destOffset + 2] = (b + (1 << 13)) >> 14; /* FIXED_FRAC_BITS */
      dest[destOffset + 1] = (g + (1 << 13)) >> 14; /* FIXED_FRAC_BITS */
      dest[destOffset] = (r + (1 << 13)) >> 14; /* FIXED_FRAC_BITS */
      destOffset = (destOffset + srcH * 4) | 0;
    }

    destOffset = ((srcY + 1) * 4) | 0;
    srcOffset = ((srcY + 1) * srcW * 4) | 0;
  }
}

// Technically, convolvers are the same. But input array and temporary
// buffer can be of different type (especially, in old browsers). So,
// keep code in separate functions to avoid deoptimizations & speed loss.

function convolveVertically(src, dest, srcW, srcH, destW, filters) {
  let r, g, b, a;
  let filterPtr, filterShift, filterSize;
  let srcPtr, srcY, destX, filterVal;
  let srcOffset = 0,
    destOffset = 0;

  // For each row
  for (srcY = 0; srcY < srcH; srcY++) {
    filterPtr = 0;

    // Apply precomputed filters to each destination row point
    for (destX = 0; destX < destW; destX++) {
      // Get the filter that determines the current output pixel.
      filterShift = filters[filterPtr++];
      filterSize = filters[filterPtr++];

      srcPtr = (srcOffset + filterShift * 4) | 0;

      r = g = b = a = 0;

      // Apply the filter to the row to get the destination pixel r, g, b, a
      for (; filterSize > 0; filterSize--) {
        filterVal = filters[filterPtr++];

        // Use reverse order to workaround deopts in old v8 (node v.10)
        // Big thanks to @mraleph (Vyacheslav Egorov) for the tip.
        a = (a + filterVal * src[srcPtr + 3]) | 0;
        b = (b + filterVal * src[srcPtr + 2]) | 0;
        g = (g + filterVal * src[srcPtr + 1]) | 0;
        r = (r + filterVal * src[srcPtr]) | 0;
        srcPtr = (srcPtr + 4) | 0;
      }

      // Bring this value back in range. All of the filter scaling factors
      // are in fixed point with FIXED_FRAC_BITS bits of fractional part.
      //
      // (!) Add 1/2 of value before clamping to get proper rounding. In other
      // case brightness loss will be noticeable if you resize image with white
      // border and place it on white background.
      //
      dest[destOffset + 3] = (a + (1 << 13)) >> 14; /* FIXED_FRAC_BITS */
      dest[destOffset + 2] = (b + (1 << 13)) >> 14; /* FIXED_FRAC_BITS */
      dest[destOffset + 1] = (g + (1 << 13)) >> 14; /* FIXED_FRAC_BITS */
      dest[destOffset] = (r + (1 << 13)) >> 14; /* FIXED_FRAC_BITS */
      destOffset = (destOffset + srcH * 4) | 0;
    }

    destOffset = ((srcY + 1) * 4) | 0;
    srcOffset = ((srcY + 1) * srcW * 4) | 0;
  }
}
function resetAlpha(dst, width, height) {
  let ptr = 3;
  const len = (width * height * 4) | 0;
  while (ptr < len) {
    dst[ptr] = 0xff;
    ptr = (ptr + 4) | 0;
  }
}
function picaResize(options) {
  const src = options.src;
  const srcW = options.width;
  const srcH = options.height;
  const destW = options.toWidth;
  const destH = options.toHeight;
  const scaleX = options.scaleX || options.toWidth / options.width;
  const scaleY = options.scaleY || options.toHeight / options.height;
  const offsetX = options.offsetX || 0;
  const offsetY = options.offsetY || 0;
  const dest = options.dest || new Uint8ClampedArray(destW * destH * 4);
  const quality = typeof options.quality === "undefined" ? 3 : options.quality;
  const alpha = options.alpha || false;

  const filtersX = createFilters(quality, srcW, destW, scaleX, offsetX),
    filtersY = createFilters(quality, srcH, destH, scaleY, offsetY);

  const tmp = new Uint8ClampedArray(destW * srcH * 4);

  // To use single function we need src & tmp of the same type.
  // But src can be CanvasPixelArray, and tmp - Uint8Array. So, keep
  // vertical and horizontal passes separately to avoid deoptimization.

  convolveHorizontally(src.data, tmp, srcW, srcH, destW, filtersX);
  convolveVertically(tmp, dest, srcH, destW, destH, filtersY);

  // That's faster than doing checks in convolver.
  // !!! Note, canvas data is not premultipled. We don't need other
  // alpha corrections.

  if (!alpha) {
    resetAlpha(dest, destW, destH);
  }
  return dest;
}

function hsv_v16(img, width, height) {
  const size = width * height;
  const out = new Uint16Array(size);
  let r, g, b, max;
  for (let i = 0; i < size; i++) {
    r = img[4 * i];
    g = img[4 * i + 1];
    b = img[4 * i + 2];
    // eslint-disable-next-line no-nested-ternary
    max = r >= g && r >= b ? r : g >= b && g >= r ? g : b;
    out[i] = max << 8;
  }
  return out;
}

function picaUnsharp(img, width, height, amount, radius, threshold) {
  let v1, v2, vmul;
  let diff, iTimes4;

  if (amount === 0 || radius < 0.5) {
    return;
  }
  if (radius > 2.0) {
    radius = 2.0;
  }

  const brightness = hsv_v16(img, width, height);

  const blured = new Uint16Array(brightness); // copy, because blur modify src

  blurMono16(blured, width, height, radius);

  const amountFp = ((amount / 100) * 0x1000 + 0.5) | 0;
  const thresholdFp = threshold << 8;

  const size = width * height;

  /* eslint-disable indent */
  for (let i = 0; i < size; i++) {
    v1 = brightness[i];
    diff = 2 * (v1 - blured[i]);

    if (Math.abs(diff) >= thresholdFp) {
      // add unsharp mask to the brightness channel
      v2 = v1 + ((amountFp * diff + 0x800) >> 12);

      // Both v1 and v2 are within [0.0 .. 255.0] (0000-FF00) range,
      // never going into[255.003 .. 255.996] (FF01-FFFF).
      // This allows to round this value as (x+.5)|0
      // later without overflowing.
      v2 = v2 > 0xff00 ? 0xff00 : v2;
      v2 = v2 < 0x0000 ? 0x0000 : v2;

      // Avoid division by 0. V=0 means rgb(0,0,0),
      // unsharp with unsharpAmount>0 cannot
      // change this value (because diff between colors gets inflated),
      // so no need to verify correctness.
      v1 = v1 !== 0 ? v1 : 1;

      // Multiplying V in HSV model by a constant is equivalent
      // to multiplying each component
      // in RGB by the same constant (same for HSL), see also:
      // https://beesbuzz.biz/code/16-hsv-color-transforms
      vmul = ((v2 << 12) / v1) | 0;

      // Result will be in [0..255] range because:
      //  - all numbers are positive
      //  - r,g,b <= (v1/256)
      //  - r,g,b,(v1/256),(v2/256) <= 255
      // So highest this number can get is X*255/X+0.5=255.5
      // which is < 256 and rounds down.

      iTimes4 = i * 4;
      img[iTimes4] = (img[iTimes4] * vmul + 0x800) >> 12; // R
      img[iTimes4 + 1] = (img[iTimes4 + 1] * vmul + 0x800) >> 12; // G
      img[iTimes4 + 2] = (img[iTimes4 + 2] * vmul + 0x800) >> 12; // B
    }
  }
}
export { picaResize, picaUnsharp };
