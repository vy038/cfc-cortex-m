#warning "cfc_math_stub.c is linked: activations are DELIBERATELY WRONG (identity, return x unchanged). This is a timing-floor build only -- it measures everything that is NOT activation math. It will fail test_golden/test_golden_nobackbone by design. Never ship this, never report its numbers as a real result."

#include "cfc_math.h"

float cfc_tanh(float x) {
    return x;
}

float cfc_sigmoid(float x) {
    return x;
}

float cfc_lecun_tanh(float x) {
    return x;
}
