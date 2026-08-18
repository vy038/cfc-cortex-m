#ifndef CFC_H
#define CFC_H

#include "weights.h"            /* generated: CFC_INPUT_DIM, CFC_HIDDEN_DIM, backbone weight arrays */
#include "weights_nobackbone.h" /* generated: CFC_NB_INPUT_DIM, CFC_NB_HIDDEN_DIM, no-backbone weight arrays */

typedef struct {
    float h[CFC_HIDDEN_DIM]; /* CFC_HIDDEN_DIM == CFC_NB_HIDDEN_DIM: shared across both configs */
} cfc_state_t;

void cfc_init(cfc_state_t *state);

/* backbone config: input -> Linear+lecun_tanh(128) -> ff1/ff2/time_a/time_b */
void cfc_step_backbone(cfc_state_t *state, const float *input, float ts, float *output);

/* no-backbone config: input concat hidden fed directly into ff1/ff2/time_a/time_b */
void cfc_step_nobackbone(cfc_state_t *state, const float *input, float ts, float *output);

#endif
