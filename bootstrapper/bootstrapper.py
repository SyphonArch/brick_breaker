import os
import evaluator
import game
import pickle
import explorer
from progressbar import ProgressBar

EVALUATOR_PATH = './evaluators'
HISTORY_PATH = './histories'


def evaluator_name(generation: int) -> str:
    """Return the name of the saved model, given generation number."""
    return f"EV-{generation}.pt"


def simulate(generation: int, iterations: int, gui=False):
    """Simulate up to the specified number of iterations. Resumes from saved files."""
    gen_path = f"{HISTORY_PATH}/gen-{generation}"
    if os.path.exists(gen_path):
        # Check how many iterations have been saved so far.
        maximum_it = -1
        it_count = 0
        with os.scandir(gen_path) as it:
            for entry in it:
                if entry.is_file():
                    if entry.name.endswith('.pickle'):
                        filename_split = entry.name.split('.')
                        assert len(filename_split) == 2
                        assert filename_split[0].isdigit()
                        iteration = int(filename_split[0])
                        maximum_it = max(maximum_it, iteration)
                        it_count += 1
        # If the following assertion fails, there are some iteration numbers missing.
        assert it_count == maximum_it + 1
        start = it_count
        print(f"Resuming from iteration {start}.")
    else:
        os.mkdir(gen_path)
        start = 0

    ev_path = f"{EVALUATOR_PATH}/{evaluator_name(generation)}"
    if generation == 0 and start == 0 and not os.path.exists(ev_path):
        # If this is the first of the first gen, a random evaluator is initialized.
        ev = evaluator.Evaluator()
        ev.save(ev_path)
    else:
        assert os.path.exists(ev_path)
        ev = evaluator.Evaluator()
        ev.load(ev_path)
    ev.eval()

    bar = ProgressBar(iterations)
    bar.i = start
    bar.start()
    for i in range(start, iterations):
        gameobj = game.main("Bricks", explorer.create_explorer(ev), block=False, fps_cap=0, gui=gui)
        assert gameobj.score != -1
        assert gameobj.score + 1 == len(gameobj.history)
        with open(f'{gen_path}/{i}.pickle', 'wb') as f:
            pickle.dump(gameobj.history, f)
        bar.update()


if __name__ == "__main__":
    simulate(0, 1000)
