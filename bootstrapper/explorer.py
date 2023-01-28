import game
import numpy as np
import constants
from multiprocessing import Pool
import os
import evaluator
from copy import deepcopy
import evaluator
import torch

RESOLUTION = 128
_seeds = np.array([i / (RESOLUTION - 1) for i in range(RESOLUTION)])
_candidates = _seeds * (constants.ANGLE_MAX_RAD - constants.ANGLE_MIN_RAD) + constants.ANGLE_MIN_RAD
CPU_COUNT = os.cpu_count()


def step_game(gamevar_angle):
    gamevar, angle = gamevar_angle
    gamevar.ai_override = lambda _: angle
    gamevar.gui = False
    gamevar.block = False
    gamevar.step()
    return evaluator.extract(gamevar)


explorer = None  # This is needed so that the local function 'explorer' can be pickled for multiprocessing


def create_explorer(network, cpu=CPU_COUNT):
    global explorer

    def explorer(gamevar: game.Game) -> float:
        to_simulate = [(deepcopy(gamevar), angle) for angle in _candidates]
        with Pool(cpu) as p:
            vectors = np.asarray(p.map(step_game, to_simulate))

        input_tensor = torch.tensor(vectors)
        return _candidates[np.argmax(network(input_tensor).detach())]

    return explorer
