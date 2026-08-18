#ifndef CFC_MATH_H
#define CFC_MATH_H

float cfc_tanh(float x);       /* standard tanh, used for ff1/ff2 */
float cfc_sigmoid(float x);    /* used for the time gate */
float cfc_lecun_tanh(float x); /* 1.7159 * tanh(0.666 * x), backbone activation ONLY */

#endif
