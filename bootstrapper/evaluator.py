import torch
import torch.nn as nn
import torch.nn.functional as torchf
import torch.optim as optim
import game
import numpy.typing as npt
import numpy as np


def extract(gamevar: game.Game) -> npt.NDArray[float]:
    return np.append(gamevar.grid_before_gen[1:-1].flatten(), gamevar.ball_count)


class Evaluator(nn.Module):
    """Given the game state (pre-generation of new bricks), evaluate how close it is to game-over.

    Input: brick layout (excluding top and bottom rows) [7x6] + ball count [1] == [43]
    Output: log((steps to game-over) + 1)

    This means that the output is always non-negative.
    A larger output signifies a 'better' game state.
    """

    def __init__(self):
        super().__init__()
        dim_x = 6
        dim_y = 9
        h1 = 24
        h2 = 12
        self.lin1 = nn.Linear(dim_x * (dim_y - 2) + 1, h1)
        self.drop1 = nn.Dropout(p=0.5)
        self.lin2 = nn.Linear(h1, h2)
        self.drop2 = nn.Dropout(p=0.5)
        self.lin3 = nn.Linear(h2, 1)

    def forward(self, x):
        x = torchf.relu(self.drop1(self.lin1(x)))
        x = torchf.relu(self.drop2(self.lin2(x)))
        x = torchf.relu(self.lin3(x))
        return x
