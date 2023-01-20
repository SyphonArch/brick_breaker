import game
import numpy as np
import constants
from multiprocessing import Pool
import os
import evaluator
from copy import deepcopy
import evaluator
import torch

RESOLUTION = 1024
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


explorer = None


def create_explorer(network):
    global explorer

    def explorer(gamevar: game.Game) -> float:
        to_simulate = [(deepcopy(gamevar), angle) for angle in _candidates]
        with Pool(CPU_COUNT) as p:
            vectors = np.asarray(p.map(step_game, to_simulate))

        input_tensor = torch.tensor(vectors, dtype=torch.float32)
        return _candidates[np.argmax(network(input_tensor).detach())]

    return explorer


ev = evaluator.Evaluator()
ev.eval()
gameobj = game.main("Bricks", create_explorer(ev), block=False, fps_cap=500)
print(gameobj.score)
