import evaluator
import game
import pickle
import explorer
from explorer_evaluator.paths import *
import numpy as np
from progressbar import ProgressBar


def simulate(generation: int, iterations: int, gui=False):
    """Simulate up to the specified number of iterations. Resumes from saved files."""
    if os.path.exists(history_path(generation)):
        # Check how many iterations have been saved so far.
        maximum_it = -1
        it_count = 0
        with os.scandir(history_path(generation)) as it:
            for entry in it:
                if entry.is_file():
                    result = history_file_re.match(entry.name)
                    if result is not None:
                        iteration = int(result.group(1))
                        maximum_it = max(maximum_it, iteration)
                        it_count += 1
        # If the following assertion fails, there are some iteration numbers missing.
        assert it_count == maximum_it + 1
        start = it_count
        print(f"Resuming from iteration {start}.")
    else:
        os.mkdir(history_path(generation))
        start = 0

    if generation == 0 and start == 0 and not os.path.exists(ev_path):
        # If this is the first of the first gen, a random evaluator is initialized.
        ev = evaluator.Evaluator()
        ev.save(evaluator_path(generation))
    else:
        assert os.path.exists(evaluator_path(generation))
        ev = evaluator.Evaluator()
        ev.load(evaluator_path(generation))
    ev.eval()

    bar = ProgressBar(iterations)
    bar.i = start
    bar.start()
    for i in range(start, iterations):
        gameobj = game.main("Bricks", True, explorer.create_explorer(ev), block=False, fps_cap=500, gui=gui)
        assert gameobj.score != -1
        assert gameobj.score + 1 == len(gameobj.history)
        with open(f'{history_path(generation)}/hist-{i}.pickle', 'wb') as f:
            assert history_file_re.match(os.path.basename(f.name)) is not None
            pickle.dump(gameobj.history, f)
        bar.update()


def generate_dataset(generation: int, train_count: int, test_count: int, validation_count: int) -> None:
    train_dataset = [[], []]
    test_dataset = [[], []]
    validation_dataset = [[], []]

    train_idx_ceiling = train_count
    test_idx_ceiling = train_count + test_count
    validation_idx_ceiling = train_count + test_count + validation_count

    # Assert that all entries are divided among the three datasets, with no leftovers.
    assert len(list(os.scandir(history_path(generation)))) == validation_idx_ceiling

    # Generate datasets from history files.
    with os.scandir(history_path(generation)) as it:
        for entry in it:
            if entry.is_file():
                result = history_file_re.match(entry.name)
                if result is not None:
                    iteration = int(result.group(1))
                    if iteration < train_idx_ceiling:
                        dataset = train_dataset
                    elif iteration < test_idx_ceiling:
                        dataset = test_dataset
                    elif iteration < validation_idx_ceiling:
                        dataset = validation_dataset
                    else:
                        raise IndexError("This shouldn't happen, given the assertion above.")

                    with open(entry.path, 'rb') as f:
                        history = pickle.load(f)
                    score = len(history) - 1  # calculate the score of the game
                    # ignore the first entry of history, as it doesn't have a pre-gen entry
                    for i in range(1, len(history)):
                        pre_gen, post_gen, ball_count = history[i]
                        steps_to_game_over = score - i
                        data = evaluator.convert(pre_gen, ball_count)
                        label = np.log(steps_to_game_over + 1, dtype=np.float32)
                        dataset[0].append(data)
                        dataset[1].append(label)
                        # As the game is symmetric, flip the grid and add as data.
                        data = evaluator.convert(np.fliplr(pre_gen), ball_count)
                        dataset[0].append(data)
                        dataset[1].append(label)

    if not os.path.exists(dataset_path(generation)):
        os.mkdir(dataset_path(generation))

    with open(f"{dataset_path(generation)}/train.pickle", 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(f"{dataset_path(generation)}/test.pickle", 'wb') as f:
        pickle.dump(test_dataset, f)
    with open(f"{dataset_path(generation)}/validation.pickle", 'wb') as f:
        pickle.dump(validation_dataset, f)


if __name__ == "__main__":
    simulate(1, 1200, gui=True)
    generate_dataset(1, 1000, 100, 100)
