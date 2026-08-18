import torch
from ncps.torch import CfCCell

INPUT_DIM = 6
HIDDEN_DIM = 32
BACKBONE_UNITS = 128
SEED = 42


def build_model(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, seed=SEED):
    torch.manual_seed(seed)
    cell = CfCCell(
        input_dim,
        hidden_dim,
        mode="default",
        backbone_activation="lecun_tanh",
        backbone_units=BACKBONE_UNITS,
        backbone_layers=1,
        backbone_dropout=0.0,
    )
    cell.eval()
    return cell


if __name__ == "__main__":
    print(build_model())
