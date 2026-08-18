import os

import torch

from export import OUT_DIR, c_array, flatten, gen_sequences, run_golden, write_golden_json
from train import HIDDEN_DIM, INPUT_DIM, SEED

# Confirmed directly from ncps 1.0.1 CfCCell.forward source:
#   x = torch.cat([input, hx], 1)
#   if self.backbone_layers > 0:
#       x = self.backbone(x)
# With backbone_layers=0 that whole branch is skipped, so x stays the raw
# concat(input, hx) and is fed straight into ff1/ff2/time_a/time_b. Matches
# __init__'s `cat_shape = hidden_size + input_size if backbone_layers == 0
# else backbone_units`.
EQUATIONS_COMMENT = """/*
 * CfC cell equations (ncps 1.0.1, ncps/torch/cfc_cell.py, mode="default", backbone_layers=0):
 *
 *   x          = concat(input, hx)          // (input_dim + hidden_dim), fed DIRECTLY into ff1/ff2/time_a/time_b
 *   ff1        = tanh(ff1_W @ x + ff1_b)
 *   ff2        = tanh(ff2_W @ x + ff2_b)
 *   t_a        = time_a_W @ x + time_a_b
 *   t_b        = time_b_W @ x + time_b_b
 *   t_interp   = sigmoid(t_a * ts + t_b)
 *   new_hidden = ff1 * (1 - t_interp) + t_interp * ff2
 *   output     = new_hidden   (CfCCell returns (new_hidden, new_hidden))
 *
 * No backbone stage: the `if self.backbone_layers > 0: x = self.backbone(x)`
 * branch in CfCCell.forward is skipped entirely when backbone_layers == 0.
 */
"""


def build_nobackbone_model(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, seed=SEED):
    from ncps.torch import CfCCell

    torch.manual_seed(seed)
    cell = CfCCell(input_dim, hidden_dim, mode="default", backbone_layers=0)
    cell.eval()
    return cell


def write_weights_nobackbone_h(cell, input_dim, hidden_dim, path):
    lines = [EQUATIONS_COMMENT]
    lines.append(f"#define CFC_NB_INPUT_DIM  {input_dim}\n")
    lines.append(f"#define CFC_NB_HIDDEN_DIM {hidden_dim}\n")
    lines.append("#define CFC_NB_CAT_DIM     (CFC_NB_INPUT_DIM + CFC_NB_HIDDEN_DIM)\n\n")

    lines.append(c_array("CFC_NB_FF1_W", flatten(cell.ff1.weight)))
    lines.append(c_array("CFC_NB_FF1_B", flatten(cell.ff1.bias)))
    lines.append(c_array("CFC_NB_FF2_W", flatten(cell.ff2.weight)))
    lines.append(c_array("CFC_NB_FF2_B", flatten(cell.ff2.bias)))
    lines.append(c_array("CFC_NB_TIME_A_W", flatten(cell.time_a.weight)))
    lines.append(c_array("CFC_NB_TIME_A_B", flatten(cell.time_a.bias)))
    lines.append(c_array("CFC_NB_TIME_B_W", flatten(cell.time_b.weight)))
    lines.append(c_array("CFC_NB_TIME_B_B", flatten(cell.time_b.bias)))

    with open(path, "w") as f:
        f.writelines(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cell = build_nobackbone_model(INPUT_DIM, HIDDEN_DIM)
    assert cell.backbone is None, "expected no backbone module when backbone_layers=0"

    write_weights_nobackbone_h(
        cell, INPUT_DIM, HIDDEN_DIM, os.path.join(OUT_DIR, "weights_nobackbone.h")
    )

    seqs = gen_sequences(INPUT_DIM)
    results = run_golden(cell, seqs, HIDDEN_DIM)
    write_golden_json(
        results, INPUT_DIM, HIDDEN_DIM, os.path.join(OUT_DIR, "golden_nobackbone.json")
    )

    print(f"wrote {os.path.join(OUT_DIR, 'weights_nobackbone.h')}")
    print(f"wrote {os.path.join(OUT_DIR, 'golden_nobackbone.json')}")


if __name__ == "__main__":
    main()
