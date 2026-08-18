import json
import math
import os

import numpy as np
import torch

from train import build_model, INPUT_DIM, HIDDEN_DIM, BACKBONE_UNITS

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "weights", "generated")
NUM_TIMESTEPS = 500

EQUATIONS_COMMENT = """/*
 * CfC cell equations (ncps 1.0.1, ncps/torch/cfc_cell.py, mode="default", with backbone):
 *
 *   x          = concat(input, hx)                       // (input_dim + hidden_dim)
 *   backbone   = lecun_tanh(backbone_W @ x + backbone_b)  // lecun_tanh(z) = 1.7159 * tanh(0.666 * z)
 *   ff1        = tanh(ff1_W @ backbone + ff1_b)           // standard tanh, NOT lecun_tanh
 *   ff2        = tanh(ff2_W @ backbone + ff2_b)           // standard tanh, NOT lecun_tanh
 *   t_a        = time_a_W @ backbone + time_a_b
 *   t_b        = time_b_W @ backbone + time_b_b
 *   t_interp   = sigmoid(t_a * ts + t_b)
 *   new_hidden = ff1 * (1 - t_interp) + t_interp * ff2
 *   output     = new_hidden   (CfCCell returns (new_hidden, new_hidden))
 *
 * NOTE: lecun_tanh is used ONLY inside the backbone activation.
 *       ff1/ff2 and the sigmoid gate use standard torch.tanh / torch.sigmoid.
 */
"""


def flatten(t):
    return t.detach().numpy().astype(np.float32).reshape(-1).tolist()


def c_array(name, values, dtype="float"):
    body = ", ".join(f"{v:.9g}f" for v in values)
    return f"static const {dtype} {name}[{len(values)}] = {{ {body} }};\n"


def write_weights_h(cell, input_dim, hidden_dim, path):
    backbone_linear = cell.backbone[0]
    lines = [EQUATIONS_COMMENT]
    lines.append(f"#define CFC_INPUT_DIM      {input_dim}\n")
    lines.append(f"#define CFC_HIDDEN_DIM     {hidden_dim}\n")
    lines.append(f"#define CFC_BACKBONE_UNITS {BACKBONE_UNITS}\n")
    lines.append("#define CFC_CAT_DIM        (CFC_INPUT_DIM + CFC_HIDDEN_DIM)\n\n")

    lines.append(c_array("CFC_BACKBONE_W", flatten(backbone_linear.weight)))
    lines.append(c_array("CFC_BACKBONE_B", flatten(backbone_linear.bias)))
    lines.append(c_array("CFC_FF1_W", flatten(cell.ff1.weight)))
    lines.append(c_array("CFC_FF1_B", flatten(cell.ff1.bias)))
    lines.append(c_array("CFC_FF2_W", flatten(cell.ff2.weight)))
    lines.append(c_array("CFC_FF2_B", flatten(cell.ff2.bias)))
    lines.append(c_array("CFC_TIME_A_W", flatten(cell.time_a.weight)))
    lines.append(c_array("CFC_TIME_A_B", flatten(cell.time_a.bias)))
    lines.append(c_array("CFC_TIME_B_W", flatten(cell.time_b.weight)))
    lines.append(c_array("CFC_TIME_B_B", flatten(cell.time_b.bias)))

    with open(path, "w") as f:
        f.writelines(lines)


def gen_sequences(input_dim, seed=42):
    rng = np.random.default_rng(seed)
    n = NUM_TIMESTEPS
    seqs = []

    # step_a: single step at t=250, two seeded levels per channel
    levels_a0 = rng.uniform(-1, 1, input_dim)
    levels_a1 = rng.uniform(-1, 1, input_dim)
    step_a = np.tile(levels_a0, (n, 1))
    step_a[250:] = levels_a1
    seqs.append(("step_a", step_a, np.ones(n, dtype=np.float32)))

    # step_b: two steps at t=150/350, three seeded levels per channel
    levels_b = rng.uniform(-1, 1, (3, input_dim))
    step_b = np.tile(levels_b[0], (n, 1))
    step_b[150:350] = levels_b[1]
    step_b[350:] = levels_b[2]
    seqs.append(("step_b", step_b, np.ones(n, dtype=np.float32)))

    # chirp_a / chirp_b: linear-sweep sine per channel, manual phase integration
    def chirp(f0, f1, amp):
        t = np.arange(n) / n
        phase = 2 * math.pi * (f0 * t + (f1 - f0) * t**2 / 2)
        return amp * np.sin(phase)

    for name in ("chirp_a", "chirp_b"):
        f0 = rng.uniform(0.5, 3.0, input_dim)
        f1 = rng.uniform(3.0, 10.0, input_dim)
        amp = rng.uniform(0.3, 1.0, input_dim)
        cols = [chirp(f0[c], f1[c], amp[c]) for c in range(input_dim)]
        seq = np.stack(cols, axis=1).astype(np.float32)
        seqs.append((name, seq, np.ones(n, dtype=np.float32)))

    # noise: gaussian, with per-timestep varying ts in [0.8, 1.2] to exercise time-gating
    noise = rng.normal(0.0, 1.0, (n, input_dim)).astype(np.float32)
    ts_noise = rng.uniform(0.8, 1.2, n).astype(np.float32)
    seqs.append(("noise", noise, ts_noise))

    return seqs


def run_golden(cell, seqs, hidden_dim):
    results = []
    with torch.no_grad():
        for name, inputs, ts in seqs:
            hx = torch.zeros(1, hidden_dim)
            hidden_states = []
            outputs = []
            for t in range(inputs.shape[0]):
                inp = torch.from_numpy(inputs[t : t + 1].astype(np.float32))
                out, hx = cell(inp, hx, float(ts[t]))
                hidden_states.append(hx.squeeze(0).numpy().tolist())
                outputs.append(out.squeeze(0).numpy().tolist())
            results.append(
                {
                    "name": name,
                    "inputs": inputs.astype(np.float64).round(9).tolist(),
                    "ts": ts.astype(np.float64).round(9).tolist(),
                    "hidden_states": hidden_states,
                    "outputs": outputs,
                }
            )
    return results


def write_golden_json(seqs_results, input_dim, hidden_dim, path):
    payload = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_timesteps": NUM_TIMESTEPS,
        "sequences": seqs_results,
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cell = build_model(INPUT_DIM, HIDDEN_DIM)

    write_weights_h(cell, INPUT_DIM, HIDDEN_DIM, os.path.join(OUT_DIR, "weights.h"))

    seqs = gen_sequences(INPUT_DIM)
    results = run_golden(cell, seqs, HIDDEN_DIM)
    write_golden_json(results, INPUT_DIM, HIDDEN_DIM, os.path.join(OUT_DIR, "golden.json"))

    print(f"wrote {os.path.join(OUT_DIR, 'weights.h')}")
    print(f"wrote {os.path.join(OUT_DIR, 'golden.json')}")


if __name__ == "__main__":
    main()
