import game
import numpy as np
import constants
from multiprocessing import Pool
import os
from typing import Callable
from copy import deepcopy
import explorer_evaluator.evaluator as evaluator
import explorer_evaluator.paths as paths
import torch

_seeds = np.array([i / (constants.EXPLORER_RESOLUTION - 1) for i in range(constants.EXPLORER_RESOLUTION)])
_candidates = _seeds * (constants.ANGLE_MAX_RAD - constants.ANGLE_MIN_RAD) + constants.ANGLE_MIN_RAD
CPU_COUNT = os.cpu_count()


def step_game(gamevar_angle):
    gamevar, angle = gamevar_angle
    gamevar.ai_function = lambda _: angle
    gamevar.ai_override = True
    gamevar.gui = False
    gamevar.block = False
    gamevar.step()
    return evaluator.extract(gamevar)


explorer = None  # This is needed so that the local function 'explorer' can be pickled for multiprocessing
hc_explorer = None


def create_explorer(network: evaluator.Evaluator, cpu=CPU_COUNT) -> Callable[[game.Game], float]:
    """Given an Evaluator network, creates an AI agent."""
    global explorer

    def explorer(gamevar: game.Game) -> float:
        to_simulate = [(deepcopy(gamevar), angle) for angle in _candidates]
        with Pool(cpu) as p:
            vectors = np.asarray(p.map(step_game, to_simulate))

        input_tensor = torch.tensor(vectors)
        network.eval()
        return _candidates[np.argmax(network(input_tensor).detach())]

    return explorer


def create_explorer_from_gen(generation: int, cpu=CPU_COUNT) -> Callable[[game.Game], float]:
    """Generate explorer from file."""
    ev = evaluator.Evaluator()
    ev.load(paths.evaluator_path(generation))
    return create_explorer(ev, cpu)


def create_hardcoded_explorer(cpu=CPU_COUNT) -> Callable[[game.Game], float]:
    """Creates a hard-coded evaluator based AI agent."""
    global hc_explorer
    grid_weights = np.asarray([[1, 1, 1, 1, 1, 1],
                               [2, 2, 2, 2, 2, 2],
                               [3, 3, 3, 3, 3, 3],
                               [4, 4, 4, 4, 4, 4],
                               [6, 6, 6, 6, 6, 6],
                               [15, 15, 15, 15, 15, 15],
                               [99999] * 6]).flatten()
    ball_count_weight = 2

    evaluation_matrix = np.append(grid_weights, -ball_count_weight)

    def hc_explorer(gamevar: game.Game) -> float:
        """Hard coded version of explorer."""
        to_simulate = [(deepcopy(gamevar), angle) for angle in _candidates]
        with Pool(cpu) as p:
            vectors = np.asarray(p.map(step_game, to_simulate))
        return _candidates[np.argmin(vectors @ evaluation_matrix)]

    return hc_explorer
