/*
 * Continued-fraction tanh, rewritten for re-measurement (the original was
 * deleted). tanh(x) = x / (1 + u/(3 + u/(5 + ... u/(15)))), u = x^2:
 * 8 vdiv.f32 per call. Build input for bench-cf only; same clamp and
 * sigmoid/lecun wrappers as the shipped Pade core.
 */
#include "cfc_math.h"

static float cf_tanhf(float x) {
    float xc = x;
    if (xc > 3.5f)
        xc = 3.5f;
    else if (xc < -3.5f)
        xc = -3.5f;
    float u = xc * xc;
    float t = 15.0f;
    t = 13.0f + u / t;
    t = 11.0f + u / t;
    t = 9.0f + u / t;
    t = 7.0f + u / t;
    t = 5.0f + u / t;
    t = 3.0f + u / t;
    return xc / (1.0f + u / t);
}

float cfc_tanh(float x) {
    return cf_tanhf(x);
}

float cfc_sigmoid(float x) {
    return 0.5f + 0.5f * cf_tanhf(0.5f * x);
}

float cfc_lecun_tanh(float x) {
    return 1.7159f * cf_tanhf(0.666f * x);
}
