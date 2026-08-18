/*
 * Original pre-optimization implementation, restored verbatim from
 * NOTES.md's record of it for the three-way pade/libm/stub timing
 * comparison. This is NOT used by src/cfc_math.c (the current, active
 * implementation is the Pade[4/3] approximation) -- it exists only as a
 * `bench-libm`/`bench-libm-nobackbone` build input so the newlib baseline
 * can be re-measured under the same caches-on conditions as everything
 * else, since every prior "libm" measurement was taken with ART caches
 * disabled (see NOTES.md's measurement history, and OPTIMIZATION.md for
 * why those early numbers were invalid).
 */
#include <math.h>

#include "cfc_math.h"

float cfc_tanh(float x) {
    return tanhf(x);
}

float cfc_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float cfc_lecun_tanh(float x) {
    return 1.7159f * tanhf(0.666f * x);
}
