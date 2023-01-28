import re

EVALUATOR_PATH = './evaluators'
HISTORY_PATH = './histories'
DATASET_PATH = './datasets'
history_file_re = re.compile(r'^hist-(\d+)\.pickle$')


def history_path(generation: int) -> str:
    """Returns the path to the generation's histories."""
    return f"{HISTORY_PATH}/gen-{generation}"


def dataset_path(generation: int) -> str:
    """Returns the path to the generation's datasets"""
    return f"{DATASET_PATH}/gen-{generation}"


def evaluator_name(generation: int) -> str:
    """Return the name of the saved model, given generation number."""
    return f"EV-{generation}.pt"


def evaluator_path(generation: int) -> str:
    """Return the name of the saved model, given generation number."""
    return f"{EVALUATOR_PATH}/{evaluator_name(generation)}"
