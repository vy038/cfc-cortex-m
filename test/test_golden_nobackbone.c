/*
 * Hand-rolled scanner tailored exactly to the golden_nobackbone.json schema
 * emitted by python/export_nobackbone.py. Not a general JSON parser: it
 * anchors on the fixed key order (name, inputs, ts, hidden_states, outputs)
 * per sequence and reads a known number of floats after each key, ignoring
 * JSON punctuation entirely. Duplicated from test_golden.c rather than
 * shared, since it exercises the fully separate cfc_step_nobackbone path.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cfc.h"

#define MAX_TIMESTEPS 500
#define MAX_SEQUENCES 5
#define ABS_ERR_LIMIT 1e-5f
#define DIVERGENCE_LIMIT 1e-6f

typedef struct {
    char name[32];
    float inputs[MAX_TIMESTEPS][CFC_NB_INPUT_DIM];
    float ts[MAX_TIMESTEPS];
    float hidden_states[MAX_TIMESTEPS][CFC_NB_HIDDEN_DIM];
    float outputs[MAX_TIMESTEPS][CFC_NB_HIDDEN_DIM];
} golden_seq_t;

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)size + 1);
    if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
        fprintf(stderr, "short read on %s\n", path);
        exit(2);
    }
    buf[size] = '\0';
    fclose(f);
    return buf;
}

static const char *find_key(const char *p, const char *key) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *loc = strstr(p, pattern);
    if (!loc) {
        fprintf(stderr, "key not found in golden_nobackbone.json: %s\n", key);
        exit(2);
    }
    const char *colon = strchr(loc, ':');
    return colon + 1;
}

static const char *skip_to_number(const char *p) {
    while (*p && !((*p >= '0' && *p <= '9') || *p == '-' || *p == '+'))
        p++;
    return p;
}

static const char *read_int(const char *p, int *out) {
    p = skip_to_number(p);
    char *endptr;
    *out = (int)strtol(p, &endptr, 10);
    return endptr;
}

static const char *read_floats(const char *p, float *out, int count) {
    for (int i = 0; i < count; i++) {
        p = skip_to_number(p);
        char *endptr;
        out[i] = strtof(p, &endptr);
        p = endptr;
    }
    return p;
}

static const char *read_string(const char *p, char *out, int max_len) {
    while (*p != '"')
        p++;
    p++;
    int i = 0;
    while (*p != '"' && i < max_len - 1)
        out[i++] = *p++;
    out[i] = '\0';
    return p + 1;
}

static const char *parse_sequence(const char *p, golden_seq_t *seq) {
    p = find_key(p, "name");
    p = read_string(p, seq->name, sizeof(seq->name));
    p = find_key(p, "inputs");
    p = read_floats(p, &seq->inputs[0][0], MAX_TIMESTEPS * CFC_NB_INPUT_DIM);
    p = find_key(p, "ts");
    p = read_floats(p, seq->ts, MAX_TIMESTEPS);
    p = find_key(p, "hidden_states");
    p = read_floats(p, &seq->hidden_states[0][0], MAX_TIMESTEPS * CFC_NB_HIDDEN_DIM);
    p = find_key(p, "outputs");
    p = read_floats(p, &seq->outputs[0][0], MAX_TIMESTEPS * CFC_NB_HIDDEN_DIM);
    return p;
}

static void check_dim(const char *name, int got, int expect) {
    if (got != expect) {
        fprintf(stderr, "golden_nobackbone.json %s mismatch: got %d, expected %d\n", name, got, expect);
        exit(2);
    }
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "../weights/generated/golden_nobackbone.json";
    char *buf = read_file(path);

    int input_dim, hidden_dim, num_timesteps;
    const char *p = find_key(buf, "input_dim");
    p = read_int(p, &input_dim);
    p = find_key(p, "hidden_dim");
    p = read_int(p, &hidden_dim);
    p = find_key(p, "num_timesteps");
    p = read_int(p, &num_timesteps);

    check_dim("input_dim", input_dim, CFC_NB_INPUT_DIM);
    check_dim("hidden_dim", hidden_dim, CFC_NB_HIDDEN_DIM);
    check_dim("num_timesteps", num_timesteps, MAX_TIMESTEPS);

    p = find_key(p, "sequences");

    static golden_seq_t seqs[MAX_SEQUENCES];
    for (int s = 0; s < MAX_SEQUENCES; s++)
        p = parse_sequence(p, &seqs[s]);

    free(buf);

    int any_fail = 0;

    for (int s = 0; s < MAX_SEQUENCES; s++) {
        golden_seq_t *seq = &seqs[s];
        cfc_state_t state;
        cfc_init(&state);

        float max_abs_err = 0.0f;
        float max_rel_err = 0.0f;
        int divergence_t = -1;

        for (int t = 0; t < MAX_TIMESTEPS; t++) {
            float output[CFC_NB_HIDDEN_DIM];
            cfc_step_nobackbone(&state, seq->inputs[t], seq->ts[t], output);

            for (int i = 0; i < CFC_NB_HIDDEN_DIM; i++) {
                float refs[2] = {seq->hidden_states[t][i], seq->outputs[t][i]};
                float gots[2] = {state.h[i], output[i]};
                for (int k = 0; k < 2; k++) {
                    float diff = fabsf(gots[k] - refs[k]);
                    float rel = diff / fmaxf(fabsf(refs[k]), 1e-8f);
                    if (diff > max_abs_err)
                        max_abs_err = diff;
                    if (rel > max_rel_err)
                        max_rel_err = rel;
                    if (divergence_t < 0 && diff > DIVERGENCE_LIMIT) {
                        divergence_t = t;
                        printf("[%s] first divergence > %g at t=%d, dim=%d, field=%s (got=%.9g, ref=%.9g)\n",
                               seq->name, DIVERGENCE_LIMIT, t, i, k == 0 ? "hidden" : "output", gots[k], refs[k]);
                    }
                }
            }
        }

        int fail = max_abs_err > ABS_ERR_LIMIT;
        any_fail |= fail;
        printf("[%s] max_abs_err=%.9g max_rel_err=%.9g %s\n", seq->name, max_abs_err, max_rel_err,
               fail ? "FAIL" : "ok");
    }

    return any_fail ? 1 : 0;
}
